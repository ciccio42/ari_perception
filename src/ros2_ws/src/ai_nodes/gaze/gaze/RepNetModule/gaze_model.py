import torch
import cv2
from pathlib import Path
from demo_utils.tensorrt import  TensorRTInterface, OnnxModel, Onnxconf
from .sixdrepnet.model import SixDRepNet
from .sixdrepnet import utils
from PIL import Image
from torchvision import transforms
import numpy as np

class RepnetTensorRT(TensorRTInterface):
    def __init__(self, fp16):
        self.transformations = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._fp16 = fp16
        self.torch_model = self._load_torch_model()
        onnx_conf = Onnxconf()
        onnx_conf.set_axes(input_names=['input'], output_names=['output'])
        onnx_model = OnnxModel(self.torch_model, 224, 224, fp16=self._fp16, onnx_conf=onnx_conf)
        super().__init__(onnx_model, dummy_inputs_batch_size=[1])
    
    def _load_torch_model(self):
        model: torch.nn.Module = SixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False)
        weights = self._load_weights()
        model.load_state_dict(weights)
        return model
    
    def _load_weights(self):
        dict_path = Path("/workspace/models/6DRepNet/weights/6DRepNet_300W_LP_AFLW2000.pth")
        assert dict_path.exists()
        saved_state_dict = torch.load(str(dict_path))
        return saved_state_dict
    
    def __call__(self, img):
        return self.infer(img)
    
    def infer(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transformations(img)
        img = img.cpu().detach().numpy()
        pred = super().__call__(img)[0]
        # pred = torch.from_numpy(pred)
        euler = utils.compute_euler_angles_from_rotation_matrices(pred)*180/np.pi
        p = euler[:, 0]
        y = euler[:, 1]
        r = euler[:, 2]

        return p,y,r
    
