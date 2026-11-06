import sys
import torch
from torch import functional as F
from torchaudio.transforms import MFCC
from pathlib import Path
from messages.msg import *
from asd.model.model.Model import ASD_Model2, ASD_Model
from demo_utils.tensorrt import  TensorRTInterface
import onnx
import onnxruntime as ort
from asd.model.loss import *
import numpy as np
from pathlib import Path
from typing import Union
from onnxconverter_common import float16
import cv2 as cv
from typing import List
from demo_utils.image_utils import ImgUtils

class Preprocessing(torch.nn.Module):
    def __init__(self, dtype, fps, rate, max_temporal_size):
        super().__init__()
        self.dim = 112
        self.dtype = dtype
        self._fps = fps
        self._rate = rate
        self.max_temporal_size = max_temporal_size
        self._mfcc_dict = self._configure_mfcc(rate, max_temporal_size)
        self._img_utils = ImgUtils()
        assert self._fps is not None
        assert self._rate is not None
    
    def _configure_mfcc(self, rate, max_temporal_size):
        mfcc_dict = {}
        numcep = 13
        melkwargs = {
            'n_fft': 1024,
            'win_length': int(0.025 * rate * 25/self._fps),
            'hop_length': int(0.01 * rate * 25/self._fps),
            'center': False
        }
        mfcc_dict[max_temporal_size] = MFCC(n_mfcc=numcep, melkwargs=melkwargs).cuda()
        dummy = torch.rand((1, 16000*2)).cuda()
        mfcc_dict[max_temporal_size](dummy)
        del dummy
        return mfcc_dict
    
    def _collect_video(self, data: List[torch.Tensor]) -> torch.Tensor:
        data_out = []
        spans = []
        for x in data:
            # x: (T, 112, 112)
            T = x.shape[0] 
            spans.append(T)
            if T < self.max_temporal_size:
                diff = self.max_temporal_size - T 
                assert diff > 0
                empty_frame = torch.zeros((diff, self.dim, self.dim), dtype=x.dtype).to(x.device)
                out_x = torch.cat([x, empty_frame], dim=0)
                data_out.append(out_x)
            else:
                data_out.append(x)
        video = torch.stack(data_out, dim=0)
        assert len(spans) == video.shape[0]
        return video, spans

    def preprocess_video(self, faces: List[List[np.ndarray]]) -> np.ndarray:
        faces = [self.preprocess_image(face) for face in faces]
        video, spans = self._collect_video(faces)
        video = video.to(self.dtype)
        return video, spans
    
    def preprocess_image(self, faces):
        faces_batch = self._img_utils.resize_and_collect(faces, self.dim, self.dim)
        faces_batch = self._img_utils.bgr_to_rgb(faces_batch) #(T,3,H,W)
        faces_batch = self._img_utils.rgb_to_gray(faces_batch) #(T,1,H,W)
        faces_batch = faces_batch.squeeze(1) #(T,H,W)
        return faces_batch
    
    def preprocess_audio(self, audio: np.ndarray, numFrames) -> np.ndarray:
        # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
        if isinstance(audio, list):
            audio = np.array(audio)
        assert isinstance(audio, np.ndarray), type(audio)
        #Input shape = (T, chunks)
        # print("Prepocess audio input:", audio.shape) 
        audio = torch.from_numpy(audio).cuda()
        audio = torch.flatten(audio)
        audio = audio.unsqueeze(0)
        mfcc = self._mfcc_dict[numFrames]
        audio = mfcc(audio)
        maxAudio = int(numFrames * 4)
        if audio.shape[0] < maxAudio:
            shortage    = maxAudio - audio.shape[0]
            audio     = F.pad(audio, (0, shortage, 0, 0), value=0)
        audio = audio.squeeze(0).transpose(0, 1)
        audio: torch.Tensor = audio[:int(round(numFrames * 4)),:]
        # print("Prepocess audio output:", audio.shape)
        #Output shape: (T, NMFCC)
        audio = audio.to(self.dtype)
        return audio
    
    def forward(self, video, audio):
        video_features, spans = self.preprocess_video(video)
        audio_features = self.preprocess_audio(audio, self.max_temporal_size) #(T, NMFCC)
        audio_features = audio_features.expand(video_features.shape[0], -1, -1) #(B,T,NMFCC)
        assert isinstance(video_features, torch.Tensor), type(video_features)
        assert isinstance(audio_features, torch.Tensor), type(audio_features)
        return video_features, audio_features, spans

class ASDModelTensorRT(TensorRTInterface):
    def __init__(self, model_threshold, fps, rate, max_temporal_size, fp16):
        self._model_threshold = model_threshold
        dtype = torch.float16 if fp16 else torch.float32
        self.fp16 = fp16
        self.dtype = dtype
        self._preprocessing = Preprocessing(self.dtype, fps, rate, max_temporal_size)
        self._model = self._load_onnx_model()
        self.ort_sess = self.build_inference_session(self._model)
    
    @property
    def model_threshold(self):
        return self._model_threshold
    
    @model_threshold.setter
    def model_threshold(self, x):
        self._model_threshold = x

    def load_model(self, model, path: str):
        selfState = model.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                name = name.replace("model.", "")
                name = name.replace("loss.", "")                                
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
    
    def _load_onnx_model(self):
        curr_dir = Path(__file__).parent
        path = curr_dir.joinpath(".cache",f"asd_model_{self._preprocessing.max_temporal_size}_.onnx")
        self.lossAV = lossAV().cpu()
        if not path.exists():
            weights_path = curr_dir.joinpath("model", "weight", "pretrain_AVA_CVPR.model")
            model = ASD_Model2()
            self.load_model(model, str(weights_path))
            model = model.eval()
            video = torch.rand(1,self._preprocessing.max_temporal_size,self._preprocessing.dim, self._preprocessing.dim)
            audio = torch.rand(1,self._preprocessing.max_temporal_size*4,13)
            input_names = ["audio", "video"]
            output_names = ["av"]
            path.parent.mkdir(parents=True, exist_ok=True)
            opset = 17
            dynamic_inputs = {
                'video': {0: 'batch'},
                'audio': {0: 'batch'},
            }
            torch.onnx.export(model, (audio,video), str(path), 
                            input_names=input_names, 
                            output_names=output_names,
                            dynamic_axes=dynamic_inputs,
                            opset_version=opset,          # explicit opset
                            dynamo=False,              # force legacy exporter  ← key fix
                            )
        model = onnx.load(str(path))
        if self.fp16:
            model = float16.convert_float_to_float16(model)
        return model
    
    def build_inference_session(self, model):
        options = [{}, {}]
        providers = ['CUDAExecutionProvider', "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        ort_sess = ort.InferenceSession(model.SerializeToString(), 
                                        providers=providers,
                                        provider_options=options,
                                        sess_options=sess_options)
        providers = ort_sess.get_providers()
        # assert 'TensorrtExecutionProvider' in providers
        return ort_sess
    
    def _asd(self, video_features, audio_features) -> Union[torch.Tensor, torch.Tensor]:
        video_features = video_features.cpu().numpy()
        audio_features = audio_features.cpu().numpy()
        probs = self.ort_sess.run(["av"], {"audio": audio_features, "video": video_features})[0]
        out = np.where(probs[:, :, 1] >= self.model_threshold, 1, 0)
        out = out.tolist()
        return out
    
    def __call__(self, video: List[np.ndarray], audio: np.ndarray):
        video_features, audio_features, spans = self._preprocessing(video, audio)
        asd_list = self._asd(video_features, audio_features)
        for i, e in enumerate(asd_list):
            asd_list[i] = e[:spans[i]]
        return asd_list
    
class ASDModel(torch.nn.Module):
    def __init__(self, model_threshold, fps, rate, max_temporal_size):
        super().__init__()
        curr_dir = Path(__file__).parent
        weights_path = curr_dir.joinpath("model", "weight", "pretrain_AVA_CVPR.model")
        self.threshold = model_threshold
        self.dtype = torch.float32
        self.model = ASD_Model().eval().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        self._preprocessing = Preprocessing(self.dtype, fps, rate, max_temporal_size)
        self.load_model(str(weights_path))
    
    def _asd(self, video_features, audio_features) -> Union[torch.Tensor, torch.Tensor]:
        # video_features: (B, T, W, H)
        #audio_features: (B, T*4, NMFCC)
        # print("Input shape video:", video_features.shape)
        # print("Input shape audio:", audio_features.shape)
        embedA = self.model.forward_audio_frontend(audio_features.cuda())
        embedV = self.model.forward_visual_frontend(video_features.cuda())	
        out = self.model.forward_audio_visual_backend(embedA, embedV)
        x = self.lossAV.FC(out)
        x = x.view(-1, video_features.shape[1], 2)
        probs = F.softmax(x, dim=-1)
        res = torch.where(probs[:, :, 1] >= self.threshold, 1, 0)
        return res
    
    def forward(self, video: List[np.ndarray], audio: np.ndarray):
        video_features, audio_features, spans = self._preprocessing(video, audio)
        asd_res = self._asd(video_features, audio_features)
        asd_list = asd_res.cpu().detach().tolist()
        for i, e in enumerate(asd_list):
            asd_list[i] = e[:spans[i]]
        return asd_list
    
    def load_model(self, path: str):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)