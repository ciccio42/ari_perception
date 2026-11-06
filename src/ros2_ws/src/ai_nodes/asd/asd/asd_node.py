import cv_bridge
from pathlib import Path
import rclpy
from demo_utils.ros_utils import wait_for_message
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import CompressedImage
from messages.msg import *
from collections import deque, OrderedDict
from asd.model.loss import *
import numpy as np
import cv2 as cv
import warnings
import datetime
import PIL
from pathlib import Path
from typing import List, Dict, Any
from asd.asd_model import ASDModelTensorRT, ASDModel
from copy import deepcopy
from copy import copy as shallow_copy
from concurrent.futures import Future
from enum import Enum
from abc import ABC
from scipy.io import wavfile
from demo_utils.ros_utils import *

warnings.filterwarnings("ignore")

class ImageBufferNotFullError(Exception): pass
class ImageBufferEmptyError(Exception): pass

def _save_img(img, idx):
    curr = Path(__file__).parent
    save_dir = curr.joinpath("test")
    #shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True)
    img = PIL.Image.fromarray(img)
    img.save(f"{save_dir}/single{idx}.jpg")
    
class InfoEnum(Enum):
    FACES = "faces"
    AUDIO = "audio"
    SEQ = "seq"
    TRACK_IDS = "track_ids"
    BOXES = "boxes"

class Info(ABC):
    
    def __init__(self, seq):
        self._info = OrderedDict()
        self._seq = seq
    
    def put(self):
        raise NotImplementedError
    
    @property
    def info(self):
        return self._info
    
    @property
    def seq(self):
        return self._seq
    
    def __str__(self):
        return self.info.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()

class InfoImage(Info):
    def __init__(self, faces, seq, track_ids: list, boxes):
        super().__init__(seq)
        self.put(faces, seq, track_ids, boxes)
    
    def put(self, faces, seq, track_ids: list, boxes):
        self._info[InfoEnum.FACES] = faces
        self._info[InfoEnum.SEQ] = seq
        self._info[InfoEnum.TRACK_IDS] = track_ids
        self._info[InfoEnum.BOXES] = boxes

class InfoAudio(Info):
    def __init__(self, seq, audio: np.ndarray):
        super().__init__(seq)
        self.put(seq, audio)
    
    def put(self, seq, audio: np.ndarray):
        self._info[InfoEnum.AUDIO] = audio
        self._info[InfoEnum.SEQ] = seq
    
    @property
    def audio(self):
        return self._info[InfoEnum.AUDIO]

class InfoData:
    def __init__(self, stamp, info: Info):
        self._info_data: Dict[int, List[Info]] = OrderedDict()
        self._stamp = stamp
        self.put(info, info)
    
    def put(self, stamp, info: Info):
        if not stamp in self._info_data:
            self._info_data[stamp] = []
        self._info_data[stamp].append(info)
    
    @property
    def info_data(self):
        return self._info_data
    
    @property
    def stamp(self):
        return self._stamp
    
    @property
    def values(self):
        return self._info_data.values()
    
    def __str__(self):
        return self.info_data.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()

def _save_audio(audio: list):
    curr = Path(__file__).parent
    save_dir = curr.joinpath("test")
    data = np.array(audio, dtype=np.float32).flatten()
    print(data.shape)
    fname = f"audio.wav"
    path = str(save_dir.joinpath(fname))
    wavfile.write(path, 16000, data)
    input('Press: ')
    
class Syncronizer:
    
    def __init__(self, record_seconds, fps, rate, chuncks_len):
        self._img_buffer_len = int(record_seconds * fps)
        self._audio_buffer_len = int((record_seconds * rate) / chuncks_len)
    
    def syncronize(self, now, img_buffer: deque, audio_buffer: deque):
        img_buffer = list(img_buffer)
        audio_buffer = list(audio_buffer)
        diff_img = self._compute_diff_array(now, img_buffer)
        diff_audio = self._compute_diff_array(now, audio_buffer)
        img_frame = self._get_data_frame(img_buffer, diff_img, self._img_buffer_len)
        audio_frame = self._get_data_frame(audio_buffer, diff_audio, self._audio_buffer_len)
        return img_frame, audio_frame
    
    def _compute_diff_array(self, now, buffer: List[InfoData]) -> np.ndarray:
        assert isinstance(buffer, list), type(buffer)
        buffer = list(sorted(buffer, key=lambda x: x.stamp))
        buffer_np = np.array([b.stamp for b in buffer])
        time_array = now - buffer_np
        return time_array

    def _get_data_frame(self, buffer: list, buffer_time_diff: np.ndarray, buffer_len: int) -> List[InfoData]:
        if len(buffer_time_diff) == 0:
            raise ImageBufferNotFullError
        end = np.argmin(buffer_time_diff)
        start = end - buffer_len
        if start < 0:
            raise ImageBufferNotFullError
        buffer_frame = buffer[start:end]
        return buffer_frame

class Reorder:

    def reoder_and_collect(self, img_buffer: List[InfoData]):
        info_list_sorted = self._reorder(img_buffer)
        tracks = self._collect(info_list_sorted)
        return tracks
    
    def _reorder(self, buffer: List[InfoData]) -> List[Info]:
        info_list: List[Info] = np.array([list(e.values) for e in buffer]).flatten().tolist()
        info_list = list(sorted(info_list, key=lambda x: x.seq))
        return info_list
    
    def _collect(self, img_buffer: List[Info]) -> OrderedDict:
        
        tracks = OrderedDict()
        for img_info in img_buffer:
            track_ids = img_info.info[InfoEnum.TRACK_IDS]
            faces = img_info.info[InfoEnum.FACES]
            boxes = img_info.info[InfoEnum.BOXES]
            if isinstance(faces, Future):
                faces = faces.result()
            assert len(faces) == len(track_ids)
            for track_id, face, box in zip(track_ids, faces, boxes):
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(face)
        return tracks

class Handler(Node):

    NODE_NAME = "asd_node"

    asd_threshold: float
    model_threshold: float
    record_seconds: int
    DEBUG: bool

    fps: float
    width: int
    height: int

    rate: int
    chuncks_len: int


    PARAM_LIST = ROS2ParamList(
        ROS2Param(name='ratio', description='ASD ratio threshold', value=0.1,  dest="asd_threshold", min=0.0, max=1.0),
        ROS2Param(name='model_threshold', description='Model threshold for ASD', value=0.76, dest="model_threshold", min=0.0, max=1.0),
        ROS2Param(name='record_seconds', description='Recording duration in seconds', value=1.0, dest="record_seconds"),
        ROS2Param(name="frequency", value=9, dest="frequency"),
        ROS2Param(name='debug/asd', description='Enable ASD debug output', dest="DEBUG", value=False),
    )
    
    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAM_LIST)
        self.get_meta()
        self.image_seq = 0
        self.audio_seq = 0
        self.max_temporal_size = int(self.record_seconds * self.fps)
        self.dtype = np.float32
        self.img_queue = deque(maxlen=60)
        self.audio_queue = deque(maxlen=60)
        self.syncronizer = Syncronizer(self.record_seconds, self.fps, self.rate, self.chuncks_len)
        self.reorder = Reorder()
        self.model = ASDModelTensorRT(self.model_threshold, self.fps, self.rate, self.max_temporal_size, fp16=True)
        self.model_threshold = shallow_copy(self.model.model_threshold)
        self.br = cv_bridge.CvBridge()
        self.sub_image = self.create_subscription(FaceDetectorFrameData, "/asd/face_info", self.image_callback, qos_profile=10)
        self.sub_audio = self.create_subscription(AudioStamped, "/audio/raw", self.audio_callback, qos_profile=10)
        self.pub_active_speaksing = self.create_publisher(AsdData, "/asd/speaking", qos_profile=10)
        self._timer = self.create_timer(1/self.frequency, self.node_callback)
        self.add_on_set_parameters_callback(self.dynamic_reconfigure)
        self.get_logger().info(f"{self.NODE_NAME} started")

    def dynamic_reconfigure(self, params: List[Parameter]):
        mapper = {"model_threshold": "model.model_threshold"}
        return default_reconfigure_with_map(self, params, self.PARAM_LIST, mapper)

    def get_meta(self):
        def set_meta_camera(msg: MetaCamera):
            self.fps = msg.fps
            self.width = msg.width
            self.height = msg.height

        def set_meta_mic(msg: MetaMic):
            self.rate = msg.sample_rate
            self.chuncks_len = msg.chunks_len
        
        status_camera, msg_camera = wait_for_message(MetaCamera, self, "/meta/camera", qos_profile=LATCH_PROFILE, time_to_wait=25)
        self.get_logger().info(f"Status camera: {status_camera}")
        status_mic, msg_mic = wait_for_message(MetaMic, self, "/meta/mic", qos_profile=LATCH_PROFILE, time_to_wait=25)
        self.get_logger().info(f"Status microphone: {status_mic}")
        if not status_camera:
            raise TimeoutError("Meta camera not received")
        if not status_mic:
            raise TimeoutError("Meta microphone not received")
        set_meta_camera(msg_camera)
        set_meta_mic(msg_mic)
        
    def _is_speaking(self, asd_list):
        is_speaking = []
        for asd_user in asd_list:
            assert isinstance(asd_user, np.ndarray)
            assert asd_user.ndim == 1
            dim = len(asd_user)
            counts = np.count_nonzero(asd_user)
            rate = counts / dim
            res = 1 if rate > self.asd_threshold else 0
            is_speaking.append(res)
        return is_speaking
    
    def node_callback(self, *args, **kwargs):
        try:
            start_sync = datetime.datetime.now()
            now = self.get_clock().now().nanoseconds
            now *= 1e-9
            img_frame, audio = self.syncronizer.syncronize(now, self.img_queue, self.audio_queue)
            end_sync = datetime.datetime.now()
            self.get_logger().debug(f"Sync time: {(end_sync-start_sync).total_seconds()}")
        except ImageBufferNotFullError:
            self.publish_message([], [])
            return
        start_reorder_and_collect = datetime.datetime.now()
        tracks = self.reorder.reoder_and_collect(img_frame)
        end_reorder_and_collect = datetime.datetime.now()
        self.get_logger().debug(f"Reorder and collect time: {(end_reorder_and_collect-start_reorder_and_collect).total_seconds()}")
        if len(tracks) == 0:
            self.publish_message([], [])
        video_faces = list(tracks.values())
        audio = self._extract_audio_by_info(audio)
        if len(video_faces) == 0:
            self.publish_message([], [])
            return
        asd_data_list = self.model(video_faces, audio)
        self.publish_message(asd_data_list, list(tracks.keys()))
    
    def publish_message(self, asd_data_list: list, track_ids: list):
        msg = AsdData()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().debug(f"{asd_data_list} - {track_ids}")
        for asd_data, track_id in zip(asd_data_list, track_ids):
            asd_label = self._compute_label(asd_data)
            msg_frame = AsdFrameData(asd=asd_label, track_id=track_id)
            msg.asd_data.append(msg_frame)
        self.pub_active_speaksing.publish(msg)
    
    def _compute_label(self, asd_data) -> int:
        unique, counts = np.unique(asd_data, return_counts=True)
        count_dict = dict(zip(unique, counts))
        count_ones = count_dict.get(1, 0)
        ratio = count_ones / len(asd_data)
        return 1 if ratio > self.asd_threshold else 0
    
    def loop(self):
        rclpy.spin(self)
    
    def _preprocess_faces(self, face: CompressedImage):
        nw_face = self.br.compressed_imgmsg_to_cv2(face)
        return nw_face
    
    def _extract_audio_by_info(self, audio: List[InfoData]) -> np.ndarray:
        audio: List[InfoAudio] = self.reorder._reorder(audio)
        audio = [a.audio for a in audio]
        if self.DEBUG:
            _save_audio(audio)
        audio_np = np.array(audio, dtype=self.dtype)
        return audio_np
    
    def _convert2array(self, msg, dtype=np.float32) -> np.ndarray:
        dim = msg.layout.dim
        size = [s.size for s in dim]
        data = msg.data
        arr = np.array(data, dtype=dtype)
        arr = np.reshape(arr, size)
        return arr
    
    def image_callback(self, msg: FaceDetectorFrameData):
        faces = msg.faces
        track_ids = msg.track_ids
        assert len(faces) == len(track_ids), f"{len(faces)} - {len(track_ids)}"
        faces = [self._preprocess_faces(face) for face in faces]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        #stamp = int(msg.header.stamp.to_sec())
        boxes = self._convert2array(msg.boxes, np.uint32)
        self.image_seq += 1
        info = InfoImage(faces, self.image_seq, track_ids, boxes)
        info_data = InfoData(stamp, info)
        self.img_queue.append(info_data)
    
    def audio_callback(self, msg: AudioStamped):
        # self.get_logger().info(f"Ciao capo ricevuto audio")
        self.audio_seq += 1
        info = InfoAudio(self.audio_seq, msg.audio.data)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        info_data = InfoData(stamp, info)
        self.audio_queue.append(info_data)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = Handler()
    node.loop()

if __name__ == "__main__":
    main()