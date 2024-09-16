# Faster inference YOLOv8 🚀

## Overview

In this repository, I offer improved inference speed utilizing Yolov8 with CPU, utilizing the power of [OpenVINO](https://github.com/openvinotoolkit/openvino) and [NumPy](https://numpy.org), across both **Object Detection** and **Segmentation** tasks. This enhancement aims to minimize prediction time while upholding high-quality results. Additionally, I wrote a [blog](https://harlystudy.com/faster-inference-yolov8-for-object-detection-and-segmentation) explaining how it works in the industry as well as the code.

Below, you'll find a quickstart guide for installation and usage. Following that, adapt the source code to align with your project requirements.

## Let's get started ✨ 
- Run the code in this repository with conda environment (or miniconda)
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
```python
python main.py --image_path="image_test/IMG_0812.JPG" --model_path="best_openvino_model/openvino_model_yolov8.xml" --output_path="debug/result"
```
- Inference yolov8 faster
```python
python main_faster.py --image_path="image_test/IMG_0812.JPG" --model_path="best_openvino_model/openvino_model_yolov8.xml" --output_path="debug/result_faster"
```
## Results
- Inference time
  
| Method           | Run 1 Time (s) | Run 2 Time (s) |
|------------------|----------------|----------------|
| openvino + torch | 10.393            | 4.290            |
| openvino + numpy | **2.847**            | **1.134**            |
 
- Segmentation result

<img style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;" src="https://raw.githubusercontent.com/Harly-1506/Faster-Inference-yolov8/main/debug/result_faster/image_with_masks.jpg?token=GHSAT0AAAAAAB4F4XQWNYF25ZJ5SBKLBRTQZSIPCWA" width="383" height="424"> 
 
- Object detection result

<img style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;" src="https://raw.githubusercontent.com/Harly-1506/Faster-Inference-yolov8/main/debug/result_faster/image_with_bbox.jpg?token=GHSAT0AAAAAAB4F4XQWFZSRFOJWDLWQ3HQ2ZSIPHWQ" width="383" height="424">

## Train your model
Visit the [Ultralytics Quickstart Guide](https://docs.ultralytics.com/quickstart/) to learn how to quickly set up and start using Ultralytics YOLO models. This comprehensive guide covers installation, basic commands, and key functionalities to help you get started with YOLOv8 for various applications. Or follow my step below
#### 1. Prepare dataset:
Datasets folder

    - train
        images
        labels
    - valid
        images
        labels
    - data.yaml


#### 2. Activate env:
```python
conda activate openvino_dev
```
#### 3. Modifly this config 
   
 - /home/harly/.config/Ultralytics/settings.yaml
   
#### 4. Run command
```python
#segment
yolo task=segment mode=train epochs=200 data=./datasets/data.yaml model=yolov8x-seg.pt imgsz=640 batch=16 patience=0 device=0

#object
yolo task=detect mode=train epochs=200 data=./datasets/data.yaml model=yolov8x.pt imgsz=640 batch=16 patience=0 device=0
```
#### 5. Export openvino format
```python
python export_model.py --model-path="best_openvino_model/best.pt"
```
- Then custom the code in this repo and using for your project


___

*Author: Harly*

*If you have any problems, please leave a message in Issues*

*Give me a star :star: if you find it useful, thanks*
