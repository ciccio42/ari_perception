import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
import numpy as np
import torchvision.transforms.functional as VF
from PIL import Image

class ImgUtils:

    def __init__(self, device="cuda") -> None:
        if torch.cuda.is_available() and device == "cuda":
            self.device = device
        else:
            self.device = "cpu"

    def _collect(self, img) -> torch.Tensor:
        first_shape = len(img[0].shape)
        if first_shape == 3:
            return torch.stack(img, dim=0)
        elif first_shape == 4:
            return torch.cat(img, dim=0)
        else:
            raise Exception("Collection Tensors Error")

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        assert isinstance(img, np.ndarray), type(img)
        img: torch.Tensor = torch.from_numpy(img).to(self.device)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = torch.permute(img, [0, -1, 1, 2]) #BCHW
        return img

    def rgb_to_bgr(self, img):
        assert isinstance(img, torch.Tensor)
        bgr_img = img[:, [2, 1, 0], ...]
        return bgr_img.to(self.device)
    
    def bgr_to_rgb(self, img):
        assert isinstance(img, torch.Tensor)
        return self.rgb_to_bgr(img).to(self.device)
    
    def rgb_to_gray(self, rgb_img):
        assert isinstance(rgb_img, torch.Tensor)
        return VF.rgb_to_grayscale(rgb_img).to(self.device)
    
    def tensor_to_numpy(self, img: torch.Tensor) -> list:
        assert len(img.shape) == 4, img.shape
        imgs = torch.permute(img, [0, 2, 3, 1])
        return imgs
    
    def resize(self, img, width, height) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = self.to_tensor(img)
        assert isinstance(img, torch.Tensor)
        img = VF.resize(img, (height, width)).to(self.device)
        assert len(img.shape) == 4, img.shape
        return img
    
    def resize_and_collect(self, img, width, height) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            img = self.resize(img, width, height)
        elif isinstance(img, list):
            img = [self.resize(i, width, height) for i in img]
            img = self._collect(img)
        elif isinstance(img, np.ndarray):
            img = self.to_tensor(img)
            img = self.resize(img, width, height)
        assert len(img.shape) == 4, img.shape
        return img.to(self.device)

if __name__ == "__main__":
    import cv2 as cv
    img = cv.imread("/workspace/ros_ws/tests/ultraface_onnx/face.jpg")
    img2 = cv.imread("/workspace/ros_ws/tests/ultraface_onnx/face.jpg")
    utils = ImgUtils()
    print("Test collect:", utils.resize_and_collect([img, img2], 300, 224).shape)
    img = utils.to_tensor(img)
    rgb = utils.bgr_to_rgb(img)
    print(rgb.shape)
    nrgb = utils.tensor_to_numpy(rgb)[0]
    print(nrgb.shape)
    cv.imwrite("rgb.jpg", nrgb)
    gray = utils.rgb_to_gray(rgb)
    ngray = utils.tensor_to_numpy(gray)[0]
    print(gray.shape)
    print(ngray.shape)
    cv.imwrite("gray.jpg", ngray)