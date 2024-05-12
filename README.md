# Faster inference YOLOv8 🚀

## Overview

In this repository, I offer improved inference speed utilizing Yolov8, utilizing the power of [OpenVINO](https://github.com/openvinotoolkit/openvino) and [NumPy](https://numpy.org), across both **Object Detection** and **Segmentation** tasks. This enhancement aims to minimize prediction time while upholding high-quality results. Additionally, I'm writing a blog explaining how it works in the industry as well as the code.

Below, you'll find a quickstart guide for installation and usage. Following that, adapt the source code to align with your project requirements.

## Let's get started
- Run conda environment 
```python
git clone https://github.com/Harly-1506/Faster-Inference-yolov8.git
cd Faster-Inference-yolov8

#conda environment 
conda create -n openvino_dev python=3.8 
# activate conda environment 
conda activate openvino_dev
# install requirements for openvino_dev
pip install -r requirements.txt
```
- Inference yolov8
```
python main.py --image_path=="/image_test/IMG_0812.JPG" --model_path="/best_openvino_model/openvino_model_yolov8.xml" --output_path=="/debug/result"
```
- Inference yolov8 faster
```python
python main_faster.py --image_path=="/image_test/IMG_0812.JPG" --model_path="/best_openvino_model/openvino_model_yolov8.xml" --output_path=="/debug/result_faster"
```
