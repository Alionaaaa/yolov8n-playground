#!/usr/bin/env python3
"""
YOLOv8n inference with ONNX Runtime C++ backend

  python3 main_cpp.py --source 0
  python3 main_cpp.py --source ../media/test.mp4
"""

import argparse
import time
import cv2
import numpy as np

import yolo_onnx
import ocsort_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../media/test.png', help='Camera index or video/image path')
    parser.add_argument('--model', type=str, default='../models/yolov8n.onnx', help='Path to ONNX model')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.7, help='NMS threshold')
    return parser.parse_args()

def yolo_to_ocsort_format(yolo_boxes):
    """Convert YOLOv8 boxes to OCSort format: [x1, y1, x2, y2, confidence, class]."""
    if len(yolo_boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    
    boxes = []
    for box in yolo_boxes:
        x1 = float(box.x1)
        y1 = float(box.y1)
        x2 = float(box.x2)
        y2 = float(box.y2)
        conf = float(box.confidence)
        cls = int(box.class_id)
        boxes.append([x1, y1, x2, y2, conf, cls])
    return np.array(boxes, dtype=np.float32)


def draw_all(image, tracks, names):
    color = (0, 255, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for trk in tracks:

        x1, y1, x2, y2 = map(int, trk[:4])
        track_id = int(trk[4])
        class_id = int(trk[5])
        track_conf = float(trk[6])
        cls_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        line1 = f"ID:{track_id} Conf:{track_conf:.2f}"
        line2 = f"{cls_name}"

        y_text = y1 - 10
        for line in [line1, line2]:
            (w, h), _ = cv2.getTextSize(line, font, 0.5, 1)
            cv2.rectangle(image, (x1, y_text - h - 2), (x1 + w + 2, y_text + 2), color, -1)
            cv2.putText(image, line, (x1 + 1, y_text), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y_text -= (h + 4)

    return image


def main():
    args = parse_args()

    engine = yolo_onnx.YoloEngine(args.model, args.conf, args.nms)
    names = engine.names
    tracker = ocsort_module.OCSort(
        det_thresh=args.conf,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False
    )

    print("ONNX Runtime C++ Engine + OCSort tracker loaded. Starting inference... (press ESC to exit)")

    # If the source is an image file
    if args.source.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Cannot open image: {args.source}")
        
        t0 = time.time()
        results = engine.process(image)
        t1 = time.time()
        tracks = tracker.update(yolo_to_ocsort_format(results.boxes))
        t2 = time.time()
  
        print(f"PREDICT: {((t1 - t0) * 1000):.1f}ms, TRACK: {((t2 - t1) * 1000):.1f}ms, Objects: {len(results.boxes)}, Tracks: {len(tracks)}")
        speed = results.speed
        print(f"Speed - Preprocess: {speed['preprocess']:.1f}ms, "
            f"Inference: {speed['inference']:.1f}ms, "
            f"Postprocess: {speed['postprocess']:.1f}ms")

        annotated = draw_all(image, tracks, names)
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
        tracks = tracker.update(yolo_to_ocsort_format(results.boxes))
        t2 = time.time()

        # Compute FPS and inference time
        current_time = time.time()
        fps_frame = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        avg_fps = 0.9 * avg_fps + 0.1 * fps_frame if avg_fps else fps_frame
        predict_ms = (t1 - t0) * 1000
        track_ms = (t2 - t1) * 1000

        info_text = f"FPS:{avg_fps:.1f} PREDICT:{predict_ms:.1f}ms TRACK:{track_ms:.1f}ms"
        print(info_text)
        speed = results.speed
        print(f"Speed - Preprocess: {speed['preprocess']:.1f}ms, "
            f"Inference: {speed['inference']:.1f}ms, "
            f"Postprocess: {speed['postprocess']:.1f}ms")

        # frame = engine.visualize(frame, results.boxes)
        frame = draw_all(frame, tracks, names)
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("YOLOv8n + OCSort Tracking", frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Inference finished")

if __name__ == "__main__":
    main()
