import rclpy
from rclpy.node import Node
from demo_utils.ros_utils import *
from std_msgs.msg import Int64MultiArray
from silero_vad.utils_vad import OnnxWrapper, get_speech_timestamps
from messages.srv import Asr
from pathlib import Path
import onnxruntime as ort
import numpy as np
from demo_utils.latency_log import mark
import time
import torch
from torch.nn import functional as F
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, AutoOptimizationConfig, ORTOptimizer
from optimum.pipelines import pipeline
from transformers import AutoProcessor
from pathlib import Path
import os
from datetime import datetime 
import traceback
import wave

NODE_NAME = "asr_node"
trt_options = {
            "trt_engine_cache_enable": True,
            "trt_fp16_enable": True,
            "trt_engine_cache_path": "/workspace/.cache_tensorrt",
            "trt_max_workspace_size": 5368709120, #5GB
            "trt_dla_enable": False
            }

class WhisperNode(Node):

    silero_vad_th: float

    SAMPLING_RATE = 16000
    WHISPER_NAME = "whisper-large-v3-turbo"
    LANGUAGES = ["italian", "english"]
    LANGUAGE_MAP = {
        "italian": "it",
        "english": "en"
    }

    ROS_PARAMETERS = ROS2ParamList(ROS2Param("silero_vad_threshold", 
                                             "silero_vad_th",
                                             "Silero VAD Threshold",
                                             min=0.0,
                                             max=1.0,
                                             value=0.5,
                                             ))

    def __init__(self):
        super().__init__(NODE_NAME)
        self.workspace = os.environ["WORKSPACE"]
        self.model_path = Path(self.workspace, "models", "whisper", self.WHISPER_NAME.split("/")[-1])
        self.vad_model_path = Path(self.workspace, "models", "silero")
        self.language = self._get_language()
        self.get_logger().info("Loading Whisper model")
        self._load_processor()
        self._load_model()
        self._load_vad_model()
        [self._dummy() for _ in range(3)]
        self.get_logger().info("ASR model loaded")
        init_parameters(self, self.ROS_PARAMETERS)
        self.add_on_set_parameters_callback(self.on_config)
        self._srv = self.create_service(Asr, "/asr/whisper", self._callback)
        self.get_logger().info(f"{NODE_NAME} started")
        
        self.save_cnt = 0
        # Create tmp directory if it doesn't exist
        self.tmp_dir = "./tmp/asr_audio"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def on_config(self, params):
        return default_reconfigure(self, params, self.ROS_PARAMETERS)

    def _load_processor(self):
        self.get_logger().info(f"Load processor started")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, sampling_rate=self.SAMPLING_RATE)
        except:
            self.get_logger().info(f"Load processor exception")
            processor = AutoProcessor.from_pretrained(self.WHISPER_NAME)
            self.model_path.mkdir(exist_ok=True, parents=True)
            processor.save_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")

        self.get_logger().info(f"Load processor ended")

    def _dummy(self):
        audio = torch.rand(16000*10).numpy()
        self.predict(audio)
    
    def _get_language(self):
        language = os.environ["DEMO_LANGUAGE"]
        language = language.lower()
        if language not in self.LANGUAGES:
            raise Exception(f"{language} not available. List of languages: {self.LANGUAGES}")
        language = self.LANGUAGE_MAP[language]
        return language

    def _load_model(self) -> ort.InferenceSession:
        self.get_logger().info(f"Load model started")
        try:
            self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                provider="CUDAExecutionProvider",
            )
        except ValueError: #Model not found
            self.get_logger().info(f"Load model exception")
            model = ORTModelForSpeechSeq2Seq.from_pretrained(self.WHISPER_NAME, export=True)
            optimization_config = AutoOptimizationConfig.O4()
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(save_dir=self.model_path, optimization_config=optimization_config)
            model.save_pretrained(self.model_path)
            self._load_model()

        self.get_logger().info(f"Load model ended")
        

    def _load_vad_model(self):
        fname = "silero_vad.onnx"
        fpath = self.vad_model_path.joinpath(fname)
        self.vad_model = OnnxWrapper(fpath.as_posix())
        x =  torch.zeros(self.SAMPLING_RATE*5, dtype=torch.float32)
        get_speech_timestamps(x, self.vad_model)
        self.get_logger().info("Silero VAD loaded")

    def predict(self, audio) -> list:
        input_ids = self.processor(audio, return_tensors="pt", sampling_rate=self.SAMPLING_RATE).input_features.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            task="transcribe",
            language=self.language,
            is_multilingual=True,
            return_dict_in_generate=False,
        )
        pred = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        self.get_logger().info(f"ASR Output: {pred}")
        return pred


    def _callback(self, req: Asr.Request, resp: Asr.Response):
        # An exception here would kill the node and leave the client waiting forever
        try:
            return self._transcribe(req)
        except Exception:
            self.get_logger().error(f"ASR request failed:\n{traceback.format_exc()}")
            return Asr.Response(text="")

    def _transcribe(self, req: Asr.Request) -> Asr.Response:
        # self.get_logger().info(f"Callback VAD with threshold {self.silero_vad_th}")
        t_in = time.perf_counter()
        audio = req.audio_features
        # layout is ignored: the client can send one dim per sample, which numpy cannot reshape
        self.get_logger().info(f"Received audio: {len(audio.data)} samples, {len(audio.layout.dim)} layout dims")
        audio = np.array(audio.data, dtype=np.float32)
        t_array = time.perf_counter()
        

        # Ensure mono
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        # Clip to valid float range
        audio = np.clip(audio, -1.0, 1.0)
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        # Save audio in tmp folder
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{self.save_cnt}.wav"
        filepath = os.path.join(self.tmp_dir, filename)
        # Save WAV file
        sample_rate = self.SAMPLING_RATE  # make sure this exists (e.g., 16000)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        self.get_logger().info(f"Saved audio to {filepath}")
        t_saved = time.perf_counter()

        t_vad = time.perf_counter()
        is_speech = True if get_speech_timestamps(audio, self.vad_model, threshold=self.silero_vad_th) else False
        t_vad_end = time.perf_counter()
        if is_speech:
            self.get_logger().info("Silero VAD: is speech boss")
            text = self.predict(audio)
        else:
            self.get_logger().info("Silero VAD: no speech")
            text = ""
        t_end = time.perf_counter()
        # durations measured on this machine: no clock synchronization needed; samples pairs this log with stt.py
        mark("whisper", "done", samples=audio.size, speech=is_speech, words=len(str(text).split()),
             array_s=round(t_array - t_in, 4), save_wav_s=round(t_saved - t_array, 4),
             vad_s=round(t_vad_end - t_vad, 4), asr_s=round(t_end - t_vad_end, 4), total_s=round(t_end - t_in, 4))
        resp = Asr.Response(text=str(text))

        return resp

    def loop(self):
        rclpy.spin(self)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = WhisperNode()
    node.loop()

if __name__ == "__main__":
    main()