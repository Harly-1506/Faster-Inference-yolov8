"""
Author: Harly
Modify day: 05052024
"""

import time
start = time.time()
import cv2
import logging
import numpy as np
from typing import Tuple
from openvino.runtime import Core
from PIL import Image
import argparse
import os
import json

from utils import ops_faster

class YOLOv8DetectorSegmenter:
    
    def __init__(self, model_path, cls_names, threshold, output_path) -> None:
        self.model_path = model_path
        self.cls_names = cls_names
        self.conf_thres = threshold
        self.model = None
        self.img_size = 640
        self.device = 'CPU'
        self.output_path = output_path
        if not os.path.exists(output_path): os.makedirs(output_path)
        format = "[%(asctime)s] [%(levelname)s] %(message)s"
        log_file_path = os.path.join(output_path, 'cropAccessoryYOLOv8.log')
        logging.basicConfig(filename = log_file_path,
                            filemode = "a", format=format, level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.info("--------------- Start cropped image accessory -----------------")
        self.logger.info(f"Time load libraries: {time.time() - start} ")

    def load_model(self):
        try:
            ie = Core()
            model_ir = ie.read_model(model=self.model_path)
            self.model = ie.compile_model(model=model_ir, device_name=self.device)
        except Exception as e: self.logger.info("Failed to load IR model ===> ", e)
        if self.model is not None: self.logger.info("Successfully loaded IR model")
    
    def postprocess(self,
        pred_boxes:np.ndarray, 
        input_hw:Tuple[int, int], 
        orig_img:np.ndarray, 
        min_conf_threshold:float = 0.25, 
        nms_iou_threshold:float = 0.5, 
        agnosting_nms:bool = False, 
        max_detections:int = 300,
        pred_masks:np.ndarray = None,
        retina_mask:bool = False
    ):
        """
        YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
            pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
            retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                            segment - segmentation polygons for each element in batch
        """
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}

        preds = ops_faster.non_max_suppression(
            (pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=len(self.cls_names),
            **nms_kwargs
        )
        results = []
        proto = np.array(pred_masks) if pred_masks is not None else None
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            if proto is None:
                pred[:, :4] = ops_faster.scale_boxes(input_hw, pred[:, :4], shape).round()
                results.append({"det": pred})
                continue
            if retina_mask:
                pred[:, :4] = ops_faster.scale_boxes(input_hw, pred[:, :4], shape).round()
                masks = ops_faster.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
                segments = [ops_faster.scale_segments(input_hw, x, shape, normalize=False) for x in ops_faster.masks2segments(masks)]
            else:
                # print(pred[:,:4])
                masks = ops_faster.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
                pred[:, :4] = ops_faster.scale_boxes(input_hw, pred[:, :4], shape).round()
                segments = [ops_faster.scale_segments(input_hw, x, shape, normalize=False) for x in ops_faster.masks2segments(masks)]
            results.append({"det": pred[:, :6], "segment": segments})
        return results
    
    def main(self, image_path, output_image):
        try:
            time_load_model = time.time()
            self.load_model()
            self.logger.info(f"[INFO] Time to load model: {time.time()-time_load_model}")

            if not os.path.exists(image_path):
                self.logger.info("[INFO] Image not found ==> ERROR \n")
                return 0 

            image = np.array(Image.open(image_path))
            image_save = image.copy()

            if image is None:
                self.logger.error("[INFO] Failed to read image ==> ERROR \n")
                return 0
            
            num_outputs = len(self.model.outputs)
            
            preprocessed_image = ops_faster.processing_image(image)
            result = self.model(preprocessed_image)
            boxes = result[self.model.output(0)]
            masks = None

            if num_outputs > 1:
                masks = result[self.model.output(1)]
                self.logger.info("[INFO] Model output shape: boxes: {}, masks: {}".format(boxes.shape, masks.shape))
            
            input_hw = preprocessed_image.shape[2:]
            detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)[0]
            bboxes_, masks = detections['det'].tolist() if len(detections['det'])>0 else [], detections['segment']
            bboxes = []

            for bbox in bboxes_:
                xmin, ymin, xmax, ymax, score, cls_id = bbox
                xmin, ymin, xmax, ymax, score, cls_id = int(xmin), int(ymin), int(xmax), int(ymax), float(score), self.cls_names[int(cls_id)]
                bboxes.append([xmin, ymin, xmax, ymax, score, cls_id])

            if len(masks) == 0:
                self.logger.info("[INFO] Length of masks {} ==> ERROR".format(len(masks)))
                self.logger.info("[INFO] There are no masks or mask more than 2 found in the image.")
                self.logger.info("[INFOR] Time taken: {}".format(time.time()-start))
       
                self.logger.info("--------------- Stopping process.---------------")
                return 0
            
            #processing output
            result_device = {}
            json_path = os.path.join(self.output_path, f"results_check_device.txt")
            scores, class_id = bboxes[0][4], bboxes[0][5]
            self.logger.info(f"[INFO] Class id: {class_id}, Scores: {scores}")
            
            #working with mask 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_mask_debug = image.copy()
            for i, mask in enumerate(masks):
                # you can get score and class_id in this line
                # scores, class_id = bboxes[i][4], bboxes[i][5]
                self.logger.info(f"Processing mask {i}")
                self.logger.info(f"Mask shape: {mask.shape}")
                mask_canvas = np.zeros_like(image_mask_debug)  

                cv2.fillPoly(mask_canvas, [mask.astype(int)], (255, 255, 255))
                contours, _ = cv2.findContours(mask_canvas[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rotated_rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rotated_rect)
                box = np.int0(box)
                cv2.drawContours(image_mask_debug, [box], 0, (0, 0, 255), 2)

                x, y, w, h = cv2.boundingRect(box)
                self.logger.info(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
                mask_path = os.path.join(self.output_path, f"mask_{i}.jpg")
                cv2.imwrite(mask_path, mask_canvas)

                #plot mask on image
                image_with_mask = image_mask_debug.copy()
                cv2.fillPoly(image_with_mask, [mask.astype(int)], (0, 255, 0))
                image_mask_debug = cv2.addWeighted(image_mask_debug, 0.5, image_with_mask, 0.5, 1)

            mask_image_path = os.path.join(self.output_path, "image_with_masks.jpg")
            self.logger.info(f"[INFO] Saving image_mask_debug to {mask_image_path}")     
            cv2.imwrite(mask_image_path, image_mask_debug)
            
            image_crop = image_save.copy()
            image_crop = image_crop[y:y+h, x:x+w]
            output_image = os.path.join(self.output_path, output_image)
            cv2.imwrite(output_image, image_crop)

            #working with bbox
            image_bbox_debug = image.copy()
            for bbox in bboxes_:
                xmin, ymin, xmax, ymax, score, cls_id = bbox
                xmin, ymin, xmax, ymax, score, cls_id = int(xmin), int(ymin), int(xmax), int(ymax), float(score), self.cls_names[int(cls_id)]
                bboxes.append([xmin, ymin, xmax, ymax, score, cls_id])
                
                cv2.rectangle(image_bbox_debug, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) 
                bbox_image_path = os.path.join(self.output_path, "image_with_bbox.jpg")
                self.logger.info(f"[INFO] Saving image_bbox_debug to {bbox_image_path}")     
                cv2.imwrite(bbox_image_path, image_bbox_debug)

            #check device is a phone or accessory
            if class_id == "phone" and scores > 0.85:
                self.logger.info("[INFO] Phone detected in the image.")
                self.logger.info("[INFOR] Time taken: {}".format(time.time()-start))
                result_device.update({"device": "phone", "confidence": scores})
                with open(json_path, "w") as f:
                    json.dump(result_device, f)

                self.logger.info("[INFOR] Time taken: {}".format(time.time()-start))
                self.logger.info("--------------- Processing completed successfully.---------------")
                return 0

            if (class_id == "airpod" or class_id == "netgrear" or class_id == "appleW") and scores > 0.90:
                if (h<1500 or w<1500): 
                    self.logger.info("[INFO] Accessory detected in the image.")
                    result_device.update({"device": "accessory", "confidence": scores})
                    with open(json_path, "w") as f:
                        json.dump(result_device, f)
                    self.logger.info("[INFOR] Time taken: {}".format(time.time()-start))
                    self.logger.info("--------------- Processing completed successfully.---------------")
                    return 0    
            else:
                self.logger.info("[INFO] Not enough condition.")
                result_device.update({"device": "phone", "confidence": scores})
                with open(json_path, "w") as f:
                    json.dump(result_device, f)
                self.logger.info("[INFOR] Time taken: {}".format(time.time()-start))
                self.logger.info("--------------- Processing completed successfully.---------------")
                return 0
            
        except Exception as e:
            print(e)
            self.logger.exception("---------------An error occurred---------------")
            self.logger.exception(e)
            return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./image_test/IMG_0812.JPG" ,type=str)
    parser.add_argument("--model_path", default="./best_openvino_model/openvino_model_yolov8.xml", type=str)
    parser.add_argument("--output_path", default="./debug/result_faster" ,type=str)
    parser.add_argument("--output_image", type=str, default="result.jpg")
    args = parser.parse_args() 

    model_faster = YOLOv8DetectorSegmenter(args.model_path, cls_names = ["airpod", "appleW", "netgrear", "phone"], threshold= 0.8, output_path=args.output_path)
    model_faster.main(args.image_path, args.output_image)
