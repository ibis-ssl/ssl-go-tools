package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/RoboCup-SSL/ssl-vision-client/pkg/vision"
	// "github.com/RoboCup-SSL/ssl-go-tools/internal/referee"
	// "github.com/RoboCup-SSL/ssl-go-tools/internal/vision"
	"google.golang.org/protobuf/proto"
	"github.com/fogleman/gg"
    "github.com/icza/mjpeg"
	"io/ioutil"
	"log"
	"math"
	"os"
	"os/signal"
	"syscall"
	"strings"
	"sort"
	"github.com/RoboCup-SSL/ssl-go-tools/pkg/persistence"
	// "github.com/pkg/errors"
	// "time"
)

var visionAddress = flag.String("visionAddress", "224.5.23.2:10006", "The multicast address of ssl-vision, default: 224.5.23.2:10006")
var fullScreen = flag.Bool("fullScreen", false, "Print the formatted message to the console, clearing the screen during print")
var noDetections = flag.Bool("noDetections", false, "Print the detection messages")
var noGeometry = flag.Bool("noGeometry", false, "Print the geometry messages")
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

			// var imageData ImageData
            // if err := json.Unmarshal(b, &imageData); err != nil {
            //     log.Fatalf("failed to unmarshal JSON: %v", err)
            // }
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
		// else if r.MessageType.Id == persistence.MessageSslRefbox2013 {
		// 	var refereeMsg referee.Referee
		// 	if err := proto.Unmarshal(r.Message, &refereeMsg); err != nil {
		// 		log.Println("Could not parse referee message:", err)
		// 		continue
		// 	}
		// 	if *extractReferee {
		// 		check(writeMessage(f, r.Timestamp, &refereeMsg))
		// 	}
		// }
	}


	// receiver := vision.NewReceiver()
	// receiver.Start(*visionAddress)
	// print("Vision receiver started")

	// printContinuous(receiver)

}

type Ball struct {
    Confidence float64 `json:"confidence"`
    X          float64 `json:"x"`
    Y          float64 `json:"y"`
    PixelX     float64 `json:"pixel_x"`
    PixelY     float64 `json:"pixel_y"`
}

type Robot struct {
    Confidence float64 `json:"confidence"`
    RobotID    int     `json:"robot_id"`
    X          float64 `json:"x"`
    Y          float64 `json:"y"`
    Orientation float64 `json:"orientation"`
    PixelX     float64 `json:"pixel_x"`
    PixelY     float64 `json:"pixel_y"`
}

// ImageData は、JSONで提供される画像情報を表します
type ImageData struct {
    FrameNumber int     `json:"frame_number"`
    TCapture    float64 `json:"t_capture"`
    TSent       float64 `json:"t_sent"`
    CameraID    int     `json:"camera_id"`
    Balls       []Ball  `json:"balls"`
    RobotsBlue  []Robot `json:"robots_blue"`
    RobotsYellow []Robot `json:"robots_yellow"`
}

func printContinuous(receiver *vision.Receiver) {
	// var odd_frames [4]ImageData
	// var even_frames [4]ImageData
	// avi, err := mjpeg.New("output_video.avi", 1280, 720, 30)
    // if err != nil {
    //     log.Fatalf("failed to create video: %v", err)
    // }
    defer createVideoFromImages("output_video.avi")

	if !*noDetections {
		receiver.ConsumeDetections = func(frame *vision.SSL_DetectionFrame) {
			robots := append(frame.RobotsBlue, frame.RobotsYellow...)
			for _, robot := range robots {
				// ssl-vision may send a NaN confidence and the json serialization can not deal with it...
				if math.IsNaN(float64(*robot.Confidence)) {
					*robot.Confidence = 0
				}
			}

			b, err := json.Marshal(frame)
            if err != nil {
                log.Fatal(err)
            }

            fmt.Println(string(b))

			// var imageData ImageData
            // if err := json.Unmarshal(b, &imageData); err != nil {
            //     log.Fatalf("failed to unmarshal JSON: %v", err)
            // }

			// if imageData.FrameNumber%2 == 1 {
            //     odd_frames[imageData.CameraID] = imageData
            // } else {
            //     even_frames[imageData.CameraID] = imageData
            // }

			// if(imageData.CameraID == 3) {
			// 	outputFileName := fmt.Sprintf("frame_%d.jpeg", imageData.FrameNumber)
			// 	if imageData.FrameNumber%2 == 0 {
			// 		createImage(odd_frames, outputFileName)
			// 	} else {
			// 		createImage(even_frames, outputFileName)
			// 	}
			// }
		}
	}
	if !*noGeometry {
		receiver.ConsumeGeometry = func(frame *vision.SSL_GeometryData) {
			b, err := json.Marshal(frame)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Print(string(b))
		}
	}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	<-signals
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
			if index < len - 1 {
				if err := avi.AddFrame(frame); err != nil {
					log.Fatalf("failed to add frame to video: %v", err)
				}
			}
			index += 1
        }
    }
}