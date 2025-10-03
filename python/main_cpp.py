#!/usr/bin/env python3
"""
YOLOv8n inference with ONNX Runtime C++ backend

  python3 main_cpp.py --source 0
  python3 main_cpp.py --source ../media/test.mp4
"""

import argparse
import time
import cv2
import yolo_onnx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../media/test.png', help='Camera index or video/image path')
    parser.add_argument('--model', type=str, default='../models/yolov8n.onnx', help='Path to ONNX model')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.7, help='NMS threshold')
    return parser.parse_args()

def main():
    args = parse_args()

    engine = yolo_onnx.YoloEngine(args.model, args.conf, args.nms)

    print("ONNX Runtime C++ Engine loaded. Starting inference... (press ESC to exit)")

    # If the source is an image file
    if args.source.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Cannot open image: {args.source}")
        
        t0 = time.time()
        results = engine.process(image)
        t1 = time.time()
  
        print(f"PREDICT: {((t1 - t0) * 1000):.1f}ms, Objects: {len(results.boxes)}")
        speed = results.speed
        print(f"Speed - Preprocess: {speed['preprocess']:.1f}ms, "
            f"Inference: {speed['inference']:.1f}ms, "
            f"Postprocess: {speed['postprocess']:.1f}ms")

        annotated = engine.visualize(image, results.boxes)
        cv2.imshow("YOLOv8n Inference cpp", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Determine if source is camera index or video path
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    prev_time = time.time()
    avg_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = engine.process(frame)
        t1 = time.time()

        annotated = engine.visualize(frame, results.boxes)

        # Compute FPS and inference time
        current_time = time.time()
        fps_frame = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        avg_fps = 0.9 * avg_fps + 0.1 * fps_frame if avg_fps else fps_frame
        predict_ms = (t1 - t0) * 1000

        # Display FPS and inference time on the frame
        cv2.putText(annotated, f"FPS:{avg_fps:.1f} PREDICT:{predict_ms:.1f}ms",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"FPS: {avg_fps:.1f}, PREDICT: {predict_ms:.1f}ms, Objects: {len(results.boxes)}")
        speed = results.speed
        print(f"Speed - Preprocess: {speed['preprocess']:.1f}ms, "
            f"Inference: {speed['inference']:.1f}ms, "
            f"Postprocess: {speed['postprocess']:.1f}ms")

        cv2.imshow("YOLOv8n ONNX Inference", annotated)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Inference finished")

if __name__ == "__main__":
    main()
