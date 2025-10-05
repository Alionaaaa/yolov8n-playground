#!/usr/bin/env python3
"""
  python3 main.py --source 0
  python3 main.py --source ../media/test.mp4
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

import ocsort_module
from utility.select_track import SelectedTrack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../media/test.png')
    return parser.parse_args()

def yolo_to_ocsort_format(yolo_boxes):
    """Convert YOLOv8 boxes to OCSort format: [x1, y1, x2, y2, confidence, class]."""
    if len(yolo_boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    boxes = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = float(box.cls[0].cpu().numpy())
        boxes.append([x1, y1, x2, y2, conf, cls])
    return np.array(boxes, dtype=np.float32)
            
def draw_all(image, tracks, names, selected=None):
    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk[:4])
        track_id = int(trk[4])
        class_id = int(trk[5])
        track_conf = float(trk[6])

        cls_name = names[class_id] if class_id in names else str(class_id)

        if selected and selected.valid and np.array_equal(trk, selected.trk):
            color = (0, 0, 255)  # Red for selected track
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for other tracks
            thickness = 2

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        line1 = f"ID:{track_id} Conf:{track_conf:.2f}"
        line2 = f"{cls_name}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_text = y1 - 10
        for line in [line1, line2]:
            (w, h), _ = cv2.getTextSize(line, font, 0.5, 1)
            cv2.rectangle(image, (x1, y_text - h - 2), (x1 + w + 2, y_text + 2), color, -1)
            cv2.putText(image, line, (x1 + 1, y_text), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y_text -= (h + 4)
    return image


def main():
    args = parse_args()

    model = YOLO("../models/yolov8n.pt")
    names = model.names
    tracker = ocsort_module.OCSort(
        det_thresh=0.2,      # detection confidence threshold
        max_age=30,          # maximum frames without update
        min_hits=3,          # minimum hits for track initialization
        iou_threshold=0.3,   # IoU threshold for association
        delta_t=3,           # 
        asso_func="iou",     # association function
        inertia=0.2,         # inertia
        use_byte=False       # use ByteTrack or not
    )
    selected = SelectedTrack()

    print("Model and tracker loaded. Starting inference... (press ESC to exit)")

    # Handle image input
    if args.source.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Cannot open image: {args.source}")
        
        t0 = time.time()
        results = model.predict(image, imgsz=640, conf=0.6, device="cpu", verbose=True)
        t1 = time.time()
        
        tracks = tracker.update(yolo_to_ocsort_format(results[0].boxes))
        t2 = time.time()

        print(f"PREDICT: {((t1 - t0) * 1000):.1f}ms, Objects: {len(results[0].boxes)}, Tracks: {len(tracks)}")

        image = draw_all(image, tracks, names)
        cv2.imshow("YOLOv8n Inference", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        
    # Determine video source (camera or file)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    
    prev_time = time.time()
    avg_fps = 0.0

    mode = 1  # 1 = simple detection, 2 = selection mode
    csrt_tracker = None
    csrt_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF

        # Switch between modes with 'w' or 's' keys
        if key == ord('w') or key == ord('s'):
            if mode == 1:
                mode = 2
            else:
                mode = 1
                # Reset selected track when switching to mode 1
                selected.trk = None
                selected.index = -1
                selected.valid = False
            csrt_tracker = None

        # Spacebar to start/stop CSRT tracking
        if key == ord(' ') and selected.valid:
            if csrt_tracker is None:
                x1, y1, x2, y2 = map(int, selected.trk[:4])
                csrt_tracker = cv2.TrackerCSRT_create()
                csrt_tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                csrt_box = (x1, y1, x2 - x1, y2 - y1)
            else:
                csrt_tracker = None
                csrt_box = None

        # Update CSRT tracker if active
        if csrt_tracker:
            ok, csrt_box = csrt_tracker.update(frame)
            if ok:
                x, y, w, h = map(int, csrt_box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            else:
                # Lost tracking of object
                csrt_tracker = None
                mode = 2
            cv2.imshow("YOLO + OCSort + CSRT", frame)
            continue

        # Run inference
        t0 = time.time()
        results = model.predict(frame, imgsz=640, conf=0.4, device="cpu", verbose=True)
        t1 = time.time()

        tracks = tracker.update(yolo_to_ocsort_format(results[0].boxes)) 
        t2 = time.time()

        # Handle track selection in mode 2
        if mode == 2:
            selected.update_if_needed(tracks)
            if key == ord('a'):
                selected.move(tracks, -1)  # Previous track
            elif key == ord('d'):
                selected.move(tracks, +1)  # Next track

        frame = draw_all(frame, tracks, names, selected if mode == 2 else None)

        # Compute FPS and inference time
        current_time = time.time()
        fps_frame = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        avg_fps = 0.9 * avg_fps + 0.1 * fps_frame if avg_fps else fps_frame
        predict_ms = (t1 - t0) * 1000
        track_ms = (t2 - t1) * 1000
        
        selected_id = int(selected.trk[4]) if selected.valid else -1
        info_text = f"FPS:{avg_fps:.1f} PREDICT:{predict_ms:.1f}ms TRACK:{track_ms:.1f}ms"
        mode_text = f"MODE:{mode} SELECT:{selected_id} "
        print(info_text)

        # Display info text
        cv2.putText(frame, info_text,
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, mode_text,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("YOLO + OCSort + CSRT", frame)

        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking finished.")

if __name__ == "__main__":
    main()