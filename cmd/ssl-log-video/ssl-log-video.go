package main

import (
	"bytes"
	"flag"
	"fmt"
	"image/jpeg"
	"log"
	"math"
	"sort"

	"github.com/RoboCup-SSL/ssl-go-tools/pkg/persistence"
	"github.com/RoboCup-SSL/ssl-vision-client/pkg/vision"
	"github.com/fogleman/gg"
	"github.com/icza/mjpeg"
	"google.golang.org/protobuf/proto"
)

var (
	logFile = flag.String("file", "", "Path to a ssl-log-recorder log file")
	output  = flag.String("output", "output_video.avi", "Destination AVI file")
)

const (
	imageWidth          = 1280
	imageHeight         = 720
	fieldWidthMM        = 12000.0
	fieldHeightMM       = 9000.0
	robotRadiusMM       = 90.0
	ballRadiusMM        = 21.5
	expectedCameras     = 4
	maxBufferedFrames   = 12
	ballTrailMaxEntries = 60
	frameRate           = 30
)

var (
	scale            = math.Min(float64(imageWidth)/fieldWidthMM, float64(imageHeight)/fieldHeightMM)
	fieldPixelWidth  = fieldWidthMM * scale
	fieldPixelHeight = fieldHeightMM * scale
	fieldOffsetX     = (float64(imageWidth) - fieldPixelWidth) / 2
	fieldOffsetY     = (float64(imageHeight) - fieldPixelHeight) / 2
)

type (
	point struct {
		x float64
		y float64
	}

	aggregatedDetection struct {
		FrameNumber uint64
		CaptureTime float64
		Balls       []*vision.SSL_DetectionBall
		Blue        []*vision.SSL_DetectionRobot
		Yellow      []*vision.SSL_DetectionRobot
	}

	frameBundle struct {
		number uint64
		frames map[int]*vision.SSL_DetectionFrame
	}

	frameCollector struct {
		expected int
		maxBuf   int
		order    []uint64
		bundles  map[uint64]*frameBundle
	}

	colorSpec struct {
		r float64
		g float64
		b float64
	}

	renderer struct {
		writer    mjpeg.AviWriter
		ballTrail []point
	}
)

func main() {
	flag.Parse()

	if *logFile == "" {
		log.Fatal("log file path must be provided with --file")
	}

	logReader, err := persistence.NewReader(*logFile)
	if err != nil {
		log.Fatalf("could not open log file: %v", err)
	}
	channel := logReader.CreateChannel()

	video, err := mjpeg.New(*output, imageWidth, imageHeight, frameRate)
	if err != nil {
		log.Fatalf("failed to create avi writer: %v", err)
	}
	defer func() {
		if closeErr := video.Close(); closeErr != nil {
			log.Printf("failed to close video: %v", closeErr)
		}
	}()

	r := newRenderer(video)
	collector := newFrameCollector(expectedCameras, maxBufferedFrames)

	for record := range channel {
		if record.MessageType.Id != persistence.MessageSslVision2014 {
			continue
		}

		var packet vision.SSL_WrapperPacket
		if err := proto.Unmarshal(record.Message, &packet); err != nil {
			log.Printf("could not parse vision message: %v", err)
			continue
		}

		frame := packet.GetDetection()
		if frame == nil || frame.CameraId == nil || frame.FrameNumber == nil {
			continue
		}

		for _, bundle := range collector.Add(frame) {
			if det := aggregateDetection(bundle); det != nil {
				if err := r.Render(det); err != nil {
					log.Printf("failed to render frame %d: %v", det.FrameNumber, err)
				}
			}
		}
	}

	for _, bundle := range collector.FlushRemaining() {
		if det := aggregateDetection(bundle); det != nil {
			if err := r.Render(det); err != nil {
				log.Printf("failed to render frame %d: %v", det.FrameNumber, err)
			}
		}
	}
}

func newFrameCollector(expected, maxBuf int) *frameCollector {
	return &frameCollector{
		expected: expected,
		maxBuf:   maxBuf,
		bundles:  make(map[uint64]*frameBundle),
	}
}

func (fc *frameCollector) Add(frame *vision.SSL_DetectionFrame) []*frameBundle {
	number := uint64(frame.GetFrameNumber())
	bundle := fc.bundles[number]
	if bundle == nil {
		bundle = &frameBundle{
			number: number,
			frames: make(map[int]*vision.SSL_DetectionFrame, fc.expected),
		}
		fc.bundles[number] = bundle
		fc.order = append(fc.order, number)
	}
	bundle.frames[int(frame.GetCameraId())] = frame

	var ready []*frameBundle
	if len(bundle.frames) == fc.expected {
		ready = append(ready, fc.flush(number))
	}

	for len(fc.order) > fc.maxBuf {
		ready = append(ready, fc.flush(fc.order[0]))
	}
	return ready
}

func (fc *frameCollector) flush(number uint64) *frameBundle {
	bundle, ok := fc.bundles[number]
	if !ok {
		return nil
	}
	delete(fc.bundles, number)
	for i, n := range fc.order {
		if n == number {
			fc.order = append(fc.order[:i], fc.order[i+1:]...)
			break
		}
	}
	return bundle
}

func (fc *frameCollector) FlushRemaining() []*frameBundle {
	var ready []*frameBundle
	for len(fc.order) > 0 {
		ready = append(ready, fc.flush(fc.order[0]))
	}
	return ready
}

func aggregateDetection(bundle *frameBundle) *aggregatedDetection {
	if bundle == nil || len(bundle.frames) == 0 {
		return nil
	}

	bestBalls := make(map[string]*vision.SSL_DetectionBall)
	bestBlue := make(map[int]*vision.SSL_DetectionRobot)
	bestYellow := make(map[int]*vision.SSL_DetectionRobot)
	var captureSum float64
	var captureCount int

	for _, frame := range bundle.frames {
		if frame == nil {
			continue
		}
		captureSum += frame.GetTCapture()
		captureCount++

		for _, ball := range frame.Balls {
			key := ballSpatialKey(ball)
			if current, ok := bestBalls[key]; !ok || ball.GetConfidence() > current.GetConfidence() {
				bestBalls[key] = ball
			}
		}

		for _, robot := range frame.RobotsBlue {
			id := int(robot.GetRobotId())
			if current, ok := bestBlue[id]; !ok || robot.GetConfidence() > current.GetConfidence() {
				bestBlue[id] = robot
			}
		}

		for _, robot := range frame.RobotsYellow {
			id := int(robot.GetRobotId())
			if current, ok := bestYellow[id]; !ok || robot.GetConfidence() > current.GetConfidence() {
				bestYellow[id] = robot
			}
		}
	}

	det := &aggregatedDetection{
		FrameNumber: bundle.number,
		CaptureTime: captureSum / math.Max(1, float64(captureCount)),
		Balls:       ballsFromMap(bestBalls),
		Blue:        robotsFromMap(bestBlue),
		Yellow:      robotsFromMap(bestYellow),
	}
	return det
}

func ballSpatialKey(ball *vision.SSL_DetectionBall) string {
	const bucket = 50.0 // millimetres
	x := math.Round(float64(ball.GetX())/bucket) * bucket
	y := math.Round(float64(ball.GetY())/bucket) * bucket
	return fmt.Sprintf("%.0f:%.0f", x, y)
}

func ballsFromMap(m map[string]*vision.SSL_DetectionBall) []*vision.SSL_DetectionBall {
	out := make([]*vision.SSL_DetectionBall, 0, len(m))
	for _, ball := range m {
		out = append(out, ball)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].GetConfidence() > out[j].GetConfidence()
	})
	return out
}

func robotsFromMap(m map[int]*vision.SSL_DetectionRobot) []*vision.SSL_DetectionRobot {
	out := make([]*vision.SSL_DetectionRobot, 0, len(m))
	for _, robot := range m {
		out = append(out, robot)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].GetRobotId() < out[j].GetRobotId()
	})
	return out
}

func newRenderer(writer mjpeg.AviWriter) *renderer {
	return &renderer{
		writer: writer,
	}
}

func (r *renderer) Render(det *aggregatedDetection) error {
	dc := gg.NewContext(imageWidth, imageHeight)
	drawField(dc)

	r.updateBallTrail(primaryBall(det.Balls))
	r.drawBallTrail(dc)

	r.drawRobots(dc, det.Blue, colorSpec{r: 0.1, g: 0.3, b: 0.95}, colorSpec{r: 1, g: 1, b: 1})
	r.drawRobots(dc, det.Yellow, colorSpec{r: 0.98, g: 0.85, b: 0.05}, colorSpec{r: 0, g: 0, b: 0})
	r.drawBalls(dc, det.Balls)
	r.drawHUD(dc, det)

	return r.writeFrame(dc)
}

func (r *renderer) writeFrame(dc *gg.Context) error {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dc.Image(), &jpeg.Options{Quality: 90}); err != nil {
		return fmt.Errorf("encode jpeg: %w", err)
	}
	return r.writer.AddFrame(buf.Bytes())
}

func (r *renderer) updateBallTrail(ball *vision.SSL_DetectionBall) {
	if ball == nil {
		return
	}
	x, y := fieldToScreen(float64(ball.GetX()), float64(ball.GetY()))
	r.ballTrail = append(r.ballTrail, point{x: x, y: y})
	if len(r.ballTrail) > ballTrailMaxEntries {
		r.ballTrail = r.ballTrail[len(r.ballTrail)-ballTrailMaxEntries:]
	}
}

func (r *renderer) drawBallTrail(dc *gg.Context) {
	if len(r.ballTrail) < 2 {
		return
	}
	for i := 1; i < len(r.ballTrail); i++ {
		alpha := float64(i) / float64(len(r.ballTrail))
		dc.SetRGBA(1, 1, 1, 0.15+0.55*alpha)
		dc.SetLineWidth(2)
		dc.DrawLine(r.ballTrail[i-1].x, r.ballTrail[i-1].y, r.ballTrail[i].x, r.ballTrail[i].y)
		dc.Stroke()
	}
}

func (r *renderer) drawRobots(dc *gg.Context, robots []*vision.SSL_DetectionRobot, body colorSpec, label colorSpec) {
	radius := robotRadiusMM * scale
	for _, robot := range robots {
		x, y := fieldToScreen(float64(robot.GetX()), float64(robot.GetY()))

		dc.SetRGB(body.r, body.g, body.b)
		dc.DrawCircle(x, y, radius)
		dc.FillPreserve()
		dc.SetRGB(1, 1, 1)
		dc.SetLineWidth(2)
		dc.Stroke()

		heading := float64(robot.GetOrientation())
		hx := x + math.Cos(heading)*radius*1.3
		hy := y - math.Sin(heading)*radius*1.3
		dc.SetLineWidth(3)
		dc.DrawLine(x, y, hx, hy)
		dc.Stroke()

		labelText := fmt.Sprintf("%d", robot.GetRobotId())
		dc.SetRGBA(0, 0, 0, 0.6)
		w, h := dc.MeasureString(labelText)
		dc.DrawRoundedRectangle(x-w/2-4, y-h/2-4, w+8, h+8, 4)
		dc.Fill()

		dc.SetRGB(label.r, label.g, label.b)
		dc.DrawStringAnchored(labelText, x, y, 0.5, 0.5)
	}
}

func (r *renderer) drawBalls(dc *gg.Context, balls []*vision.SSL_DetectionBall) {
	if len(balls) == 0 {
		return
	}
	radius := ballRadiusMM * scale
	for idx, ball := range balls {
		x, y := fieldToScreen(float64(ball.GetX()), float64(ball.GetY()))
		dc.SetRGB(0.95, 0.5, 0.12)
		dc.DrawCircle(x, y, radius)
		dc.FillPreserve()
		dc.SetRGB(1, 1, 1)
		dc.SetLineWidth(2)
		dc.Stroke()

		label := fmt.Sprintf("Ball %d  conf %.2f  (%.2f m, %.2f m)", idx+1, ball.GetConfidence(), float64(ball.GetX())/1000, float64(ball.GetY())/1000)
		dc.SetRGBA(0, 0, 0, 0.6)
		w, h := dc.MeasureString(label)
		margin := 6.0
		dc.DrawRoundedRectangle(x+radius+10, y-h/2-margin, w+2*margin, h+2*margin, 4)
		dc.Fill()
		dc.SetRGB(1, 1, 1)
		dc.DrawStringAnchored(label, x+radius+10+margin, y, 0, 0.5)
	}
}

func (r *renderer) drawHUD(dc *gg.Context, det *aggregatedDetection) {
	const (
		padding   = 14.0
		lineSpace = 22.0
	)
	width := 290.0
	height := 120.0
	dc.SetRGBA(0, 0, 0, 0.55)
	dc.DrawRoundedRectangle(20, 20, width, height, 10)
	dc.Fill()

	lines := []string{
		fmt.Sprintf("Frame #%d", det.FrameNumber),
		fmt.Sprintf("Capture: %.3fs", det.CaptureTime),
		fmt.Sprintf("Balls: %d", len(det.Balls)),
		fmt.Sprintf("Blue Robots: %d", len(det.Blue)),
		fmt.Sprintf("Yellow Robots: %d", len(det.Yellow)),
	}

	for i, line := range lines {
		dc.SetRGB(1, 1, 1)
		dc.DrawStringAnchored(line, 20+padding, 20+padding+float64(i)*lineSpace, 0, 0.5)
	}
}

func primaryBall(balls []*vision.SSL_DetectionBall) *vision.SSL_DetectionBall {
	if len(balls) == 0 {
		return nil
	}
	return balls[0]
}

func drawField(dc *gg.Context) {
	background := colorSpec{r: 0.02, g: 0.24, b: 0.02}
	field := colorSpec{r: 0.1, g: 0.5, b: 0.1}
	lines := colorSpec{r: 0.95, g: 0.95, b: 0.95}

	dc.SetRGB(background.r, background.g, background.b)
	dc.Clear()

	dc.SetRGB(field.r, field.g, field.b)
	dc.DrawRectangle(fieldOffsetX, fieldOffsetY, fieldPixelWidth, fieldPixelHeight)
	dc.Fill()

	dc.SetRGB(lines.r, lines.g, lines.b)
	dc.SetLineWidth(2)
	dc.DrawRectangle(fieldOffsetX, fieldOffsetY, fieldPixelWidth, fieldPixelHeight)
	dc.Stroke()

	centerX, centerY := fieldToScreen(0, 0)
	dc.DrawLine(fieldOffsetX, centerY, fieldOffsetX+fieldPixelWidth, centerY)
	dc.Stroke()

	dc.DrawCircle(centerX, centerY, 500*scale)
	dc.Stroke()

	drawPenaltyAreas(dc)
}

func drawPenaltyAreas(dc *gg.Context) {
	const (
		penaltyDepthMM = 1000.0
		penaltyWidthMM = 2000.0
		cornerRadius   = 8.0
	)

	leftX := -fieldWidthMM/2 + penaltyDepthMM
	rightX := fieldWidthMM/2 - penaltyDepthMM
	top := penaltyWidthMM / 2

	x, y := fieldToScreen(leftX, top)
	dc.DrawRoundedRectangle(x-penaltyDepthMM*scale, y-penaltyWidthMM*scale, penaltyDepthMM*scale, penaltyWidthMM*scale*2, cornerRadius)
	dc.Stroke()

	x, y = fieldToScreen(rightX, top)
	dc.DrawRoundedRectangle(x, y-penaltyWidthMM*scale, penaltyDepthMM*scale, penaltyWidthMM*scale*2, cornerRadius)
	dc.Stroke()
}

func fieldToScreen(x, y float64) (float64, float64) {
	return fieldOffsetX + fieldPixelWidth/2 + x*scale, fieldOffsetY + fieldPixelHeight/2 + y*scale
}
