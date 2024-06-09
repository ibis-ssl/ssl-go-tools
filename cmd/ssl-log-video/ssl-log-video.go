package main

import (
	"flag"
	"fmt"
	"github.com/RoboCup-SSL/ssl-vision-client/pkg/vision"
	"github.com/RoboCup-SSL/ssl-go-tools/pkg/persistence"
	"google.golang.org/protobuf/proto"
	"github.com/fogleman/gg"
    "github.com/icza/mjpeg"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"sort"
)

var logFile = flag.String("file", "/home/hans/Downloads/STOP_AVOID_BALL_test_avoid_ball.log", "The log file to play")

func main() {
	flag.Parse()

	logReader, err := persistence.NewReader(*logFile)
	if err != nil {
		log.Fatalf("Could not create logFile reader")
	}

	channel := logReader.CreateChannel()

	var odd_frames [4]vision.SSL_DetectionFrame
	var even_frames [4]vision.SSL_DetectionFrame

	defer createVideoFromImages("output_video.avi")

	for r := range channel {
		if r.MessageType.Id == persistence.MessageSslVision2014 {
			var visionMsg vision.SSL_WrapperPacket
			if err := proto.Unmarshal(r.Message, &visionMsg); err != nil {
				log.Println("Could not parse vision wrapper message:", err)
				continue
			}
			log.Println(visionMsg)

			frame := visionMsg.Detection

			if *frame.FrameNumber%2 == 1 {
                odd_frames[*frame.CameraId] = *frame
            } else {
                even_frames[*frame.CameraId] = *frame
            }
			if(*frame.CameraId == 3) {
				outputFileName := fmt.Sprintf("frame_%d.jpeg", *frame.FrameNumber)
				if *frame.FrameNumber%2 == 0 {
					createImage(odd_frames, outputFileName)
				} else {
					createImage(even_frames, outputFileName)
				}
			}

		}
	}
}

func createImage(frame [4]vision.SSL_DetectionFrame, outputFileName string) {
    const imageWidth = 1280
    const imageHeight = 720
	const scale = imageWidth / 9000.0
    dc := gg.NewContext(imageWidth, imageHeight)

    // 背景色を設定
    dc.SetRGB(0, 1, 0) // 緑色
    dc.Clear()

	for i := 0; i < 4; i++ {
		data := frame[i]
		// ボールの位置を描画
		dc.SetRGB(1, 0.5, 0) // オレンジ色
		for _, ball := range data.Balls {
			dc.DrawCircle(float64(ball.GetX()) * scale +float64(imageWidth)/2, float64(ball.GetY()) * scale +float64(imageHeight)/2, 20 * scale)
			dc.Fill()
		}

		// 青ロボットの位置を描画
		for _, robot := range data.RobotsBlue {
			dc.SetRGB(0, 0, 1) // 青色
			dc.DrawCircle(float64(robot.GetX()) * scale +float64(imageWidth)/2, float64(robot.GetY()) * scale +float64(imageHeight)/2, 90 * scale)
			fmt.Printf("%f, %f\n", float64(robot.GetX()) * scale +float64(imageWidth)/2, float64(robot.GetY()) * scale +float64(imageHeight)/2)
			dc.Fill()
			dc.SetRGB(1, 1, 1) // 白色
			dc.DrawStringAnchored(fmt.Sprintf("%d", robot.RobotId), float64(robot.GetX()) * scale +float64(imageWidth)/2, float64(robot.GetY()) * scale +float64(imageHeight)/2, 0.5, 0.5)
		}

		// 黄色ロボットの位置を描画
		for _, robot := range data.RobotsYellow {
			dc.SetRGB(1, 1, 0) // 黄色
			dc.DrawCircle(float64(robot.GetX()) * scale +float64(imageWidth)/2, float64(robot.GetY()) * scale +float64(imageHeight)/2, 90 * scale)
			dc.Fill()
			dc.SetRGB(0, 0, 0) // 黒色
			dc.DrawStringAnchored(fmt.Sprintf("%d", robot.RobotId), float64(robot.GetX()) * scale +float64(imageWidth)/2, float64(robot.GetY()) * scale +float64(imageHeight)/2, 0.5, 0.5)
		}
	}

    // 画像を保存
	gg.SaveJPG(outputFileName, dc.Image(), 80)
    // dc.SaveJPG(outputFileName,80)
    fmt.Println("Image saved as", outputFileName)
}

func createVideoFromImages(outputFileName string){
	avi, err := mjpeg.New(outputFileName, 1280, 720, 30)
    if err != nil {
        log.Fatalf("failed to create video: %v", err)
    }
    defer avi.Close()

	// フォルダ内にある"frame_%d.png"の画像を読み込み
	files, err := ioutil.ReadDir("./")
    if err != nil {
        log.Fatalf("failed to read image file: %v", err)
    }

	sort.Slice(files, func(i, j int) bool {
        return files[i].Name() < files[j].Name()
    })

	len := len(files)
	index := 0

    for _, file := range files {
        if strings.HasPrefix(file.Name(), "frame_") {
            frame, err := ioutil.ReadFile(file.Name())
            if err != nil {
                log.Fatalf("failed to read image file: %v", err)
            }
            // 画像をビデオに追加
			if index < len {
				if err := avi.AddFrame(frame); err != nil {
					log.Fatalf("failed to add frame to video: %v", err)
				}
			}
			index += 1
        }
    }
}