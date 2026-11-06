#!/usr/bin/python3

import onnxruntime as ort
import numpy as np
from pathlib import Path
from demo_utils.tensorrt import OnnxModel, TensorRTInterface

class UltraFaceTensorRT(TensorRTInterface):
    def __init__(self, fp16=False):
        model_path = self._get_model_path()
        model = OnnxModel(model_path, 320, 240, fp16=fp16)
        super().__init__(model)
        self._image_mean = np.array([127, 127, 127])
    
    def _get_model_path(self):
        curr_dir = Path(__file__).parent
        model_name = "version-RFB-320.onnx"
        model_path = str(curr_dir.joinpath(model_name))
        return model_path
    
    def __call__(self, image):
        scores, boxes = super().__call__(image)
        return scores, boxes
    
    @property
    def net_width(self):
        return self._net_width
    
    @property
    def net_height(self):
        return self._net_height
    
    @property
    def net_mean(self):
        return self._image_mean

if __name__ == "__main__":
    net = UltraFaceTensorRT(fp16=True)