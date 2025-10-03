#!/usr/bin/env python3
"""
  python3 main.py --source 0
  python3 main.py --source ../media/test.mp4
"""

import argparse
import time
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../media/test.png')
    return parser.parse_args()

def main():
    args = parse_args()

    model = YOLO("../models/yolov8n.pt")
    print("Model loaded. Starting inference... (press ESC to exit)")


    if args.source.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Cannot open image: {args.source}")
        
        t0 = time.time()
        results = model.predict(image, imgsz=640, conf=0.6, device="cpu", verbose=True)
        t1 = time.time()
        
        print(f"PREDICT: {((t1 - t0) * 1000):.1f}ms, Objects: {len(results[0].boxes)}")
        
        annotated = results[0].plot()
        cv2.imshow("YOLOv8n Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        

    # determine video source (camera or file)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    # load YOLOv8n model
    

    prev_time = time.time()
    avg_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run inference
        t0 = time.time()
        results = model.predict(frame, imgsz=640, conf=0.6, device="cpu", verbose=True)
        t1 = time.time()

        r = results[0]
        annotated = r.plot()

        # compute FPS and inference time
        current_time = time.time()
        fps_frame = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        avg_fps = 0.9 * avg_fps + 0.1 * fps_frame if avg_fps else fps_frame

        predict_ms = (t1 - t0) * 1000
        cv2.putText(annotated, f"FPS:{avg_fps:.1f} PREDICT:{predict_ms:.1f}ms",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"FPS: {avg_fps:.1f}, PREDICT: {predict_ms:.1f}ms, Objects: {len(r.boxes)}")

        cv2.imshow("YOLOv8n Inference", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference finished.")

if __name__ == "__main__":
    main()
