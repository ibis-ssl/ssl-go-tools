package persistence

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"github.com/pkg/errors"
	"os"
)

type Writer struct {
	file       *os.File
	writer     *bufio.Writer
	gzipWriter *gzip.Writer
	Open       bool
}

func NewWriter(filename string) (logWriter Writer, err error) {
	logWriter.Open = false
	logWriter.file, err = os.Create(filename)
	if err != nil {
		err = errors.Wrap(err, "Could not create log file: "+filename)
		return
	}

	if filename[len(filename)-2:] == "gz" {
		logWriter.gzipWriter = gzip.NewWriter(logWriter.file)
		logWriter.writer = bufio.NewWriter(logWriter.gzipWriter)
	} else {
		logWriter.writer = bufio.NewWriter(logWriter.file)
	}
	err = logWriter.writeHeader()
	if err != nil {
		err = errors.Wrap(err, "Could not write header")
		return
	}
	logWriter.Open = true
	return
}

func (l *Writer) writeHeader() error {
	_, err := l.writer.WriteString("SSL_LOG_FILE")
	if err != nil {
		return err
	}
	err = l.writeInt32(1)
	return err
}

func (l *Writer) Close() error {
	if l.writer == nil {
		// not open
		return nil
	}
	err := l.writer.Flush()
	if err != nil {
		return err
	}
	if l.gzipWriter != nil {
		err = l.gzipWriter.Close()
		if err != nil {
			return err
		}
	}
	return l.file.Close()
}

func (l *Writer) Write(msg *Message) (err error) {
	err = l.writeInt64(msg.Timestamp)
	if err != nil {
		return
	}
	err = l.writeInt32(int32(msg.MessageType.Id))
	if err != nil {
		return
	}
	err = l.writeInt32(int32(len(msg.Message)))
	if err != nil {
		return
	}
	err = l.writeBytes(msg.Message)
	if err != nil {
		return
	}
	return
}

func (l *Writer) writeBytes(data []byte) error {
	_, err := l.writer.Write(data)
	return err
}

func (l *Writer) writeString(data string) error {
	_, err := l.writer.WriteString(data)
	return err
}

func (l *Writer) writeInt32(data int32) error {
	err := binary.Write(l.writer, binary.BigEndian, data)
	return err
}

func (l *Writer) writeInt64(data int64) error {
	err := binary.Write(l.writer, binary.BigEndian, data)
	return err
}
