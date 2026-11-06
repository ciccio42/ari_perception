import torch
import numpy as np
import onnx
from onnx2torch import convert
import onnxruntime as ort
from onnxconverter_common import float16
from pathlib import Path

class AxisNoneError(Exception): pass

class Onnxconf:
    
    def __init__(self):
        self._input_names = None
        self._output_names = None
        self._dynamic_axes = None
    
    def set_axes(self, input_names, output_names, dynamic_axes=None):
        self._input_names = input_names
        self._output_names = output_names
        self._dynamic_axes = dynamic_axes
    
    def _check_axis(self, axis):
        if axis is None:
            raise AxisNoneError()
    
    @property
    def input_names(self):
        self._check_axis(self._input_names)
        return self._input_names
    
    @property
    def output_names(self):
        self._check_axis(self._output_names)
        return self._output_names
    
    @property
    def dynamic_axes(self):
        return self._dynamic_axes

class OnnxModel:
    def __init__(self, model_data, input_width, input_height, channels=3, fp16=False, onnx_conf=None, prefix="", opset=None):
        self._model_data = model_data
        self._prefix = prefix
        self._net_height = input_height
        self._net_width = input_width
        self._channels = channels
        self._fp16 = fp16
        self._opset = opset
        self._onnx_model: onnx.ModelProto = self._load_model(model_data, fp16, onnx_conf)
        
    def _load_model(self, model_data, fp16, onnx_conf) -> onnx.ModelProto:
        if isinstance(model_data, torch.nn.Module):
            model_data = model_data.cuda()
            model_path = self._build_model(model_data, onnx_conf)
            model = self._read_model(model_path, fp16)
        elif isinstance(str(model_data), str):
            model = self._read_model(model_data, fp16)
        else:
            raise Exception(f"{type(model)} unknown")
        return model
    
    def _get_paths(self, model_torch):
        curr_dir = Path(__file__).parent
        models_dir = curr_dir.joinpath("..", "..", "models", "cache_onnx")
        ext = ".onnx"
        model_stem = model_torch.__class__.__name__
        model_name = self._prefix + model_stem + ext
        model_path = models_dir.joinpath(model_stem, model_name)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        return model_path
    
    def _is_model_exists(self, model_path):
        return Path(model_path).exists()
    
    def _build_model(self, model_torch: torch.nn.Module, onnx_conf: Onnxconf):
        
        def _export():
            torch.onnx.export(
            model_torch,
            dummy_input,
            model_path,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self._opset
            )
        
        if onnx_conf is None:
            raise Exception("You need build Onnxconf object to build onnx model")
        
        model_path = self._get_paths(model_torch)
        if self._is_model_exists(model_path):
            return model_path
        print(f"Building onnx model: {model_path}")
        dummy_input = torch.rand(1, 3, self.net_height, self.net_width, device="cuda")
        input_names = onnx_conf.input_names
        output_names = onnx_conf.output_names
        dynamic_axes = onnx_conf.dynamic_axes
        try:
            _export()
        except RuntimeError:
            dummy_input = torch.rand(1, self.net_height, self.net_width, 3, device="cuda")
            _export()
        return model_path
        
    def _read_model(self, model_path, fp16) -> onnx.ModelProto:
        print(f"Loading model: {model_path}")
        model = onnx.load(model_path)
        print("Onnx model loaded")
        if fp16:
            print("Converting model to FP16")
            model = float16.convert_float_to_float16(model)
            print("Model converted")
        return model
    
    @property
    def input_axis_name(self):
        names = self._onnx_model.graph.input
        assert len(names) == 1, f"{len(names)}\n\n{names}"
        name = names[0].name
        return name
    
    @property
    def net_height(self):
        return self._net_height
    
    @property
    def net_width(self):
        return self._net_width
    
    @property
    def net_channels(self):
        return self._channels
    
    @property
    def fp16(self):
        return self._fp16
    
    @property
    def onnx_model(self):
        return self._onnx_model
    
class TensorRTInterface:
    
    def __init__(self, model_onnx: OnnxModel, dummy_inputs_batch_size=[1, 2, 3], dla=False):
        self._net_height = model_onnx.net_height
        self._net_width = model_onnx.net_width
        self._channels = model_onnx.net_channels
        self._input_axis_name = model_onnx.input_axis_name
        self._fp16 = model_onnx.fp16
        self._dla = dla
        try:
            print(f"Creating inference session")
            self.ort_sess = self.build_inference_session(model_onnx.onnx_model)
            print(f"Created inference session")
        except Exception:
            print("Changing opset")
            model_onnx = self._change_opset(model_onnx)
            self.ort_sess = self.build_inference_session(model_onnx.onnx_model)
        self._dummy_inputs_batch_size = dummy_inputs_batch_size
        self.dummy_inputs(self.ort_sess)
    
    def _change_opset(self, model_onnx: OnnxModel) -> OnnxModel:
        onnx_conf = Onnxconf()
        onnx_conf.set_axes(input_names=['input'], output_names=['output'])
        torch_graph = convert(model_onnx.onnx_model)
        nw_model_onnx = OnnxModel(torch_graph, self.net_width, self.net_height,
                                  onnx_conf=onnx_conf, fp16=self._fp16, opset=8, prefix=model_onnx._prefix)
        return nw_model_onnx
    
    def __call__(self, batch):
        assert isinstance(batch, np.ndarray), f"Get instead: {type(batch)}"
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)
        if self._fp16:
            batch = batch.astype(np.float16)
        else:
            batch = batch.astype(np.float32)
        return self.ort_sess.run(None, {self._input_axis_name: batch})
    
    @property
    def net_height(self):
        return self._net_height
    
    @property
    def net_width(self):
        return self._net_width
    
    @property
    def input_axis_name(self):
        return self._input_axis_name
    
    def build_inference_session(self, model):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options = [self.build_tensorrt_options(), {}, {}]
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', "CPUExecutionProvider"]
        ort_sess = ort.InferenceSession(model.SerializeToString(), 
                                        providers=providers,
                                        provider_options=options,
                                        sess_options=sess_options)
        options = ort_sess.get_provider_options()
        providers = ort_sess.get_providers()
        return ort_sess
    
    def build_tensorrt_options(self):
        return {
            "trt_engine_cache_enable": True,
            "trt_fp16_enable": True,
            "trt_engine_cache_path": "/workspace/.cache_tensorrt",
            "trt_max_workspace_size": 5368709120, #5GB
            "trt_dla_enable": self._dla,
        }
    
    def dummy_inputs(self, ort_sess):
        batch_size = self._dummy_inputs_batch_size
        dtype = np.float16 if self._fp16 else np.float32
        inputs = [np.full((x, self._channels, self._net_height, self._net_width), 0, dtype=dtype) for x in batch_size]
        try:
            for x in inputs:
                ort_sess.run(None, {self._input_axis_name: x})
        except:
            inputs = [np.full((x, self._net_height, self._net_width, self._channels), 0, dtype=dtype) for x in batch_size]
            for x in inputs:
                ort_sess.run(None, {self._input_axis_name: x})