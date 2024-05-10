from ultralytics import YOLO

model_path = './best_openvino_model/openvino_model_yolov8.pt'
model = YOLO(model_path)
model.export(format='openvino')