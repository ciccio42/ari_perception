import cv_bridge
from pathlib import Path
import rclpy
from rclpy.node import Node
from demo_utils.ros_utils import wait_for_message
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import Image, CompressedImage
from messages.msg import *
from asd.ultraface import UltraFaceDetector
from asd.model.loss import *
import numpy as np
import warnings
import PIL
from pathlib import Path
from collections import deque
from typing import List
from asd.tracker_module import TrackerSort
from collections import OrderedDict, deque
from demo_utils.ros_utils import *

warnings.filterwarnings("ignore")

def _save_img(img):
        curr = Path(__file__).parent
        save_dir = curr.joinpath("test")
        #shutil.rmtree(save_dir)
        save_dir.mkdir(exist_ok=True)
        img = PIL.Image.fromarray(img)
        img.save(f"{save_dir}/single.jpg")

class NoFaceError(Exception): pass
class NoTrackError(Exception): pass

class FaceExcecutor:

    fps: float
    width: int
    height: int

    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.logger = self.node.get_logger()
        self.get_meta()
        self._tracker = TrackerSort(max_age=self.fps*2)
        self.face_detector = UltraFaceDetector(orig_width=self.width,
                                            orig_height=self.height,
                                            confidence_threshold=0.95,
                                            fp16=True)
        self.logger.info("Face detector loaded")
    
    def get_meta(self):
        def set_meta_camera(msg: MetaCamera):
            self.fps = msg.fps
            self.width = msg.width
            self.height = msg.height

        status, msg_camera = wait_for_message(MetaCamera, self.node, "/meta/camera", qos_profile=LATCH_PROFILE, time_to_wait=25)
        if not status:
            raise Exception("Meta camera not received")
        set_meta_camera(msg_camera)
    
    def _face_detection(self, img):
        res = self.face_detector(img) 
        info_data = self._compute_info_data(img, res)
        return info_data
    
    def _tracking(self, info: dict):
        tracks = self._tracker._update_tracks(info['box'], info['prob'])
        boxes = []
        track_ids = []
        for track in tracks:
            boxes.append(track[:4])
            track_ids.append(track[-1])
        return boxes, track_ids
    
    def _compute_info_data(self, img, info: dict):
        boxes, track_ids = self._tracking(info)
        
        faces = []
        for box in boxes:
            assert isinstance(box, np.ndarray)
            box = box.astype(np.uint32)
            assert len(box) == 4, f"{len(box)} - {type(box)}"
            face = self.face_detector._cut_image(img, box.tolist())
            faces.append(face)
        
        info["track_ids"] = track_ids
        info['box'] = boxes
        info['face'] = faces
        assert len(faces) == len(boxes) == len(track_ids), f"{len(faces)}\n{boxes}\n{track_ids}"
        return info
    
    def _order_tracks(self, info_data: dict) -> OrderedDict:
        support = OrderedDict()
        tracks = info_data['track_ids']
        for k, v in info_data.items():
            sorted_data = [x for _, x in list(sorted(zip(tracks, v), key=lambda x: x[0]))]
            support[k] = sorted_data
        assert support.keys() == info_data.keys()
        return support
    
    def __call__(self, video):
        info_data = self._face_detection(video)
        info_data = self._order_tracks(info_data)
        assert len(info_data['face']) == len(info_data["track_ids"])
        return info_data

class Handler(Node):

    NODE_NAME = "face_detection_node"

    _max_num_faces: int
    _bb_area_th: int
    frequency: int
    DEBUG: bool

    PARAM_LIST = [
        ROS2Param("max_num_faces", value=3, dest="_max_num_faces"),
        ROS2Param("bb_area_th", value=6300, dest="_bb_area_th"),
        ROS2Param("frequency", value=10, dest="frequency"),
        ROS2Param(name='debug/face_detection', value=False, dest="DEBUG")
    ]

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAM_LIST)
        self.add_on_set_parameters_callback(self._change_params)
        self._is_configured = False
        self._buffer =  deque(maxlen=1)
        self.get_logger().info(f"Face Executor")
        self.face_executor = FaceExcecutor(self)
        self.logger = self.get_logger()
        self.br = cv_bridge.CvBridge()
        self.get_logger().info(f"Creating topics CompressedImage - FaceDetectorFrameData")
        self.pub_debug = self.create_publisher(CompressedImage, "/debug/face_detection/img/compressed", 10)
        self.pub_face_detecor_data = self.create_publisher(FaceDetectorFrameData, "/asd/face_info", 10)
        self.sub_data_sync = self.create_subscription(CompressedImageStamped, "/img/compressed", self.image_callback, 10)
        self._timer = self.create_timer(1/self.frequency, self._process)

    def _change_params(self, params):
        config = parameter_list_to_dict(params)
        if not self._is_configured:
            self._is_configured = True
            config["num_faces"] = self._max_num_faces
            config["bb_area_th"] = self._bb_area_th
        else:
            self._max_num_faces = config["num_faces"]
            self._bb_area_th = config["_bb_area_th"]
            self.logger.warn(f"Changing - max num faces: {self._max_num_faces} - Area BB: {self._bb_area_th}")
        result = SetParametersResult(successful=True)
        return result

    def loop(self):
        rclpy.spin(self)
    
    def _msg_face_detector(self, msg_frames: List[FaceDetectorFrameData], audio):
        msg = FaceDetectorData(data=msg_frames, audio=audio)
        header = Header(stamp=self.get_clock().now().to_msg())
        msg.header = header
        return msg
    
    def _msg_image(self, face):
        try:
            imgmsg = self.br.cv2_to_compressed_imgmsg(face)
        except:
            return None
        return imgmsg
    
    def _msg_float_array(self, data):
        data = np.array(data, dtype=np.float32)
        data_np_shape = data.shape
        dim = [MultiArrayDimension(size=d) for d in data_np_shape]
        msg = Float32MultiArray()
        msg.data = data.flatten().tolist()
        msg.layout.dim = dim
        return msg
    
    def _filter_num_face(self, info: dict):
        def _compute_area(box):
            width = abs(box[0] - box[2])
            height = abs(box[1] - box[3])
            return width * height

        boxes = info["box"] 
        boxes = np.array(boxes)
        boxes = np.reshape(boxes, (-1, 4))
        if len(boxes) == 0:
            return info
        area_boxes = np.apply_along_axis(_compute_area, axis=1, arr=boxes)
        self.logger.debug(f"area: {area_boxes}")
        area_mask = area_boxes >= self._bb_area_th
        sort_index = np.argsort(area_boxes)
        sort_index = np.flip(sort_index)
        sort_index_filter = []

        for index in sort_index:
            if area_mask[index]:
                sort_index_filter.append(index)

        sort_index_filter = sort_index_filter[:self._max_num_faces]
    
        info["face"] = [info["face"][i] for i in sort_index_filter]
        info["box"] = [info["box"][i] for i in sort_index_filter]
        info["prob"] = [info["prob"][i] for i in sort_index_filter]
        info["track_ids"] = [info["track_ids"][i] for i in sort_index_filter]

        assert len(info["face"]) == len(info["box"]) == len(info["prob"]) == len(info["track_ids"])
        assert len(info["face"]) <= self._max_num_faces
        return info

    def _msg_frame_data(self, info_data: dict) -> FaceDetectorFrameData:
        info_data = self._filter_num_face(info_data)
        msg = FaceDetectorFrameData()
        msg.boxes = self._msg_float_array(info_data['box'])
        faces = []
        for face in info_data['face']:
            face_data = self._msg_image(face)
            if face_data is not None:
                faces.append(face_data)
        info_data['track_ids'] = info_data['track_ids'][:len(faces)]
        msg.faces = faces
        msg.probs = self._msg_float_array(info_data['prob'])
        track_ids = np.array(info_data['track_ids'], dtype=np.uint32).flatten().tolist()
        msg.track_ids = track_ids
        msg.header.stamp = self.get_clock().now().to_msg()
        return msg
    
    def _process(self, *args, **kwargs):
        if len(self._buffer) != 0:
            msg = self._buffer.pop()
            frame = self.br.compressed_imgmsg_to_cv2(msg.image)
            if self.DEBUG:
                self.pub_debug.publish(msg.image)
            info_data: dict = self.face_executor(frame)
            self.logger.debug(f"Faces detected: {len(info_data['face'])}")
            msg = self._msg_frame_data(info_data)
            self.pub_face_detecor_data.publish(msg)

    def image_callback(self, msg: CompressedImageStamped):
        self._buffer.append(msg)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = Handler()
    node.loop()

if __name__ == "__main__":
    main()