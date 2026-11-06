from demo_utils.tensorrt import OnnxModel, TensorRTInterface
from demo_utils.image_utils import ImgUtils
from pathlib import Path
from object_detector.utils import *
from object_detector.results import Results
from copy import deepcopy
from demo_utils.object_detector.coco_labels import COCO

class YOLOv8TensorRT(TensorRTInterface):
    def __init__(self, model_name= "yolov8s.onnx"):
        self._img_utils = ImgUtils()
        self._size = 640
        curr_dir = Path(__file__).parent
        model_path = Path("/workspace/models/YOLOv8") / model_name
        print(f"Loading model path {model_path}")
        model_onnx = OnnxModel(model_data=str(model_path), fp16=True, input_width=640, input_height=640)
        self._labels = COCO().labels
        print(f"Model Object Detector Loaded")
        super().__init__(model_onnx, dummy_inputs_batch_size=[1], dla=False)
        print(f"TensorRTInterfaceDone")
    
    def preprocess(self, img):
        img = self._img_utils.resize_and_collect(img, width=self._size, height=self._size)
        img = self._img_utils.bgr_to_rgb(img)
        img = img.float() / 255
        img = img.cpu().numpy()
        return img
    
    def postprocess(self, orig_img: np.ndarray, img: np.ndarray, embs: np.ndarray):
        assert isinstance(embs, np.ndarray), type(embs)
        assert isinstance(orig_img, np.ndarray), type(orig_img)
        assert isinstance(img, np.ndarray)
        assert len(orig_img.shape) == 3, orig_img.shape
        assert len(img.shape) == 4, img.shape
        assert orig_img.shape[-1] == 3, orig_img.shape
        assert img.shape[1] == 3, img.shape
        embs = torch.from_numpy(embs)
        assert len(embs.shape) == 3, embs.shape
        preds = non_max_suppression(
            embs,
            conf_thres=0.25,
            iou_thres=0.7,
            agnostic=False,
            max_det=300,
            classes=None)
        pred = preds[0]
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results = Results(orig_img, "NA", names="Yolov8", boxes=pred)
        predictions = results.boxes
        classes = predictions.cls
        confidences = predictions.conf
        boxes = predictions.xyxy
        classes = classes.cpu().numpy().flatten().astype(np.uint32).tolist()
        names = [self._labels[c] for c in classes]
        return classes, confidences, boxes, names

    def __call__(self, img):
        orig_img = deepcopy(img)
        img: np.ndarray = self.preprocess(img)
        embs = super().__call__(img)[0]
        results = self.postprocess(orig_img, img, embs)
        return results
