import numpy as np
import torch
import cv2 as cv
import asd.UltrafaceOnnx.box_utils_numpy as box_utils
from asd.UltrafaceOnnx.ultraface_onnx import UltraFaceTensorRT
from functools import partial
from pathlib import Path
import PIL


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class UltraFaceDetector:
    def __init__(self, orig_width, orig_height, confidence_threshold=-1, area_threshold=-1, fp16=False):
        self._img_width = orig_width
        self._img_height = orig_height
        self._margin = 10
        self._fp16 = fp16
        self.detector = UltraFaceTensorRT(fp16=self._fp16)
        self.confidence_threshold = confidence_threshold
        self.area_threshold = area_threshold
        
    def _preprocess(self, x: np.array):
        image = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.detector.net_width, self.detector.net_height))
        image = (image - self.detector.net_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32)
        return image
    
    def __call__(self, *args, **kwds):
        return self.detect_faces(*args, **kwds)
    
    def _cut_image(self, img, box):
        face = img[box[1]: box[3], box[0]: box[2]]
        # self._save_img(face)
        return face
    
    def _save_img(self, img):
        curr = Path(__file__).parent
        save_dir = curr.joinpath("test_img")
        #shutil.rmtree(save_dir)
        save_dir.mkdir(exist_ok=True)
        img = PIL.Image.fromarray(img)
        img.save(f"{save_dir}/single.jpg")
    
    def detect_faces(self, img):
        frame = self._preprocess(img)
        scores, boxes = self.detector(frame)
        assert scores.shape[0] == boxes.shape[0] == 1
        func = partial(
            predict, self._img_width, self._img_height, scores, boxes, 0.7
        )
        res = func()
        
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        boxes, labels, probs = res
        info = { "face": [], "box": [], "prob": [] }
        for box in boxes:
            info["box"].append(box)
            info["prob"].extend(probs)
        return info