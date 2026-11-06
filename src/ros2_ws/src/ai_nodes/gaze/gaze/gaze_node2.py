from os import stat
import numpy as np
import rclpy
from rclpy.node import Node
import cv_bridge
from datetime import datetime
from std_msgs.msg import Float32MultiArray, Header
from messages.msg import *
from collections import deque
from gaze.RepNetModule.gaze_model import RepnetTensorRT
from demo_utils.ros_utils import *
from demo_utils.ros_utils import wait_for_message

import warnings
warnings.filterwarnings("ignore")

class NoFaceDetectedError(Exception): pass

class Node(Node):
    frequency: int
    yaw_threshold: int

    fps: float
    width: int
    height: int

    rate: int
    chuncks_len: int
    
    NODE_NAME = "gaze_estimation_node"

    PARAMETERS = [
        ROS2Param("yaw_threshold", value = 26, dest = "yaw_threshold"),
        ROS2Param("frequency", value = 8, dest = "frequency")
    ]

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAMETERS)
        self.model = RepnetTensorRT(fp16=True)
        self.get_meta()
        self.data_queue = deque(maxlen=1)
        self.br = cv_bridge.CvBridge()
        self._pred_buffer = deque(maxlen=1)
        self.sub_image = self.create_subscription(FaceDetectorFrameData, "/asd/face_info", self.image_callback, 10)
        self.pub_is_looking = self.create_publisher(GazeData, "/gaze/is_looking", 10)
        self.create_timer(1.0 / self.frequency, self.node_callback)
        self.get_logger().info("Gaze model loaded")

    def node_callback(self, *args, **kwargs):
        if len(self.data_queue) == 0:
            return
        last_faces, last_boxes = self.data_queue.pop()
        yaw_angles_per_face = self.detect(last_faces)
        is_looking_mask = np.logical_and(yaw_angles_per_face > -self.yaw_threshold, yaw_angles_per_face < self.yaw_threshold)
        is_looking = np.where(is_looking_mask, 1, 0)
        last_boxes = last_boxes[:len(is_looking)]
        self.publish_message(is_looking, last_boxes, yaw_angles_per_face)
    
    def detect(self, imgs: list) -> np.ndarray:
        
        res = [self._infer(img) for img in imgs]
        
        res = np.array(res)
        return res
    
    def _infer(self, img):
        start = datetime.now()
        p,y,r  = self.model(img)
        end = datetime.now()
        self.get_logger().debug(f"Gaze inference time: {(end-start).total_seconds()}")
        return y
    
    def publish_message(self, is_looking: np.ndarray, last_boxes, yaw_angles_per_face):
        assert len(is_looking) == len(last_boxes) == len(yaw_angles_per_face), f"{is_looking} - {last_boxes}"
        is_looking = np.array(is_looking).astype(np.uint8)
        last_boxes = np.array(last_boxes).astype(np.uint32)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        msg = GazeData(header=header)
        for e, box, yaw in zip(is_looking, last_boxes, yaw_angles_per_face):
            e = e.item()
            box = box.tolist()
            inner_msg = GazeFrameData(is_looking=e, box=box, yaw=float(yaw))
            msg.gaze_data.append(inner_msg)
        self.pub_is_looking.publish(msg)
    
    def loop(self):
        rclpy.spin(self)
                
    def _convert2data(self, msg: FaceDetectorFrameData):
        frames_data = msg
        boxes = frames_data.boxes
        last_faces_msg = frames_data.faces
        last_faces = [self._convert2img(face_msg) for face_msg in last_faces_msg]
        last_boxes_msg = boxes
        last_boxes = self._convert2array(last_boxes_msg)
        if len(last_boxes) == 0:
            raise NoFaceDetectedError
        assert last_boxes.ndim == 2, last_boxes.shape
        return last_faces, last_boxes
    
    def _convert2img(self, face):
        return self.br.compressed_imgmsg_to_cv2(face)
    
    def _convert2array(self, msg: Float32MultiArray) -> np.ndarray:
        dim = msg.layout.dim
        size = [s.size for s in dim]
        data = msg.data
        arr = np.array(data, dtype=np.float32)
        arr = np.reshape(arr, size)
        return arr
    
    def image_callback(self, msg: FaceDetectorFrameData):
        try:
            last_faces, last_boxes = self._convert2data(msg)
            
        except NoFaceDetectedError: 
            self.publish_message([], [], [])
            return
        self.data_queue.append((last_faces, last_boxes))
    
    def get_meta(self):
        def set_meta_camera(msg: MetaCamera):
            self.fps = msg.fps
            self.width = msg.width
            self.height = msg.height

        def set_meta_mic(msg: MetaMic):
            self.rate = msg.sample_rate
            self.chuncks_len = msg.chunks_len
        
        status_camera, msg_camera = wait_for_message(MetaCamera, self, "/meta/camera", qos_profile=LATCH_PROFILE, time_to_wait=25)
        status_mic, msg_mic = wait_for_message(MetaMic, self, "/meta/mic", qos_profile=LATCH_PROFILE, time_to_wait=25)
        if not status_camera:
            raise TimeoutError("Meta camera not received")
        if not status_mic:
            raise TimeoutError("Meta microphone not received")
        set_meta_camera(msg_camera)
        set_meta_mic(msg_mic)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = Node()
    node.loop()

if __name__ == "__main__":
    main()
    