from demo_utils.tensorrt import TensorRTInterface, OnnxModel
from demo_utils.soft_biometrics import *
from demo_utils.image_utils import ImgUtils
from pathlib import Path
import numpy as np
import cv2 as cv


class MTNSoftBiometrics(TensorRTInterface):
    def __init__(self, fp16=False):
        assert fp16 == False, "Model with fp16 not works"
        self._img_utils = ImgUtils()
        self.gen = [x for x in GENDER]
        #age
        self.eth = [x for x  in ETHNICITY]
        self.emo = [x for x  in EMOTION]
        self._tasks = {GENDER: self.gen, AGE: None, ETHNICITY: self.eth, EMOTION: self.emo}

        curr_dir = Path(__file__).parent
        model_path = Path("/workspace/models/pepper_models/video/face_analysis/resnet50_mtn_verA.onnx")
        if not model_path.exists():
            raise Exception(f"Model: resnet50_mtn_verA.onnx not found")
        model_onnx = OnnxModel(str(model_path), input_width=112, input_height=112, fp16=fp16)
        super().__init__(model_onnx, dummy_inputs_batch_size=[1])
        dummy = np.full((3,112,112,3), 0, dtype=np.uint8)
        self.__call__(dummy)

    def __call__(self, images: list):
        return self.infer(images)

    def infer(self, images):
        images = self._preprocessing(images)
        images = images.cpu().numpy()
        batch_size = images.shape[0]
        results = super().__call__(images)
        info_list = []
        for i in range(batch_size):
            new_results = []
            for r  in results:
                new_results.append(r[i])
            info = self._postprocessing(new_results)
            info_list.append(info)
        return info_list

    def _preprocessing(self, faces: list):
        faces = self._img_utils.resize_and_collect(faces, 112, 112)
        faces = self._img_utils.bgr_to_rgb(faces)
        faces = faces.float()
        assert len(faces.shape) == 4, faces.shape
        faces[:, 0, ...] -= 93.5940
        faces[:, 1, ...] -= 104.7624
        faces[:, 2, ...] -= 129.1863
        faces = self._img_utils.tensor_to_numpy(faces)
        return faces
    
    def _postprocessing(self, results: list) -> SoftBiometricsInfo:
        info = SoftBiometricsInfo()
        for task, res in zip(self._tasks.keys(), results):

            if task == AGE:
                age = res.item()
                # print("age:", age)
                info.age = age
                continue

            res = res.flatten()
            idx = np.argmax(res)
            mapper = self._tasks[task]
            row_value = mapper[idx]
            value = task(row_value)
            if task == GENDER:
                info.gender = value
            
            elif task == EMOTION:
                info.emotion = value
            
            elif task == ETHNICITY:
                info.ethnicity = value
            
            else:
                raise Exception
        return info
