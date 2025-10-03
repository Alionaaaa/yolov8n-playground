from ultralytics import YOLO

# Load model
model = YOLO("./yolov8n.pt")
# Export to ONNX
model.export(format="onnx", imgsz=640, simplify=True)