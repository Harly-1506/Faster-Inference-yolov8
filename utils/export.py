import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Export YOLO model to OpenVINO format.')
parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model file.')
args = parser.parse_args()


model = YOLO(args.model_path)
model.export(format='openvino')