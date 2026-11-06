import rclpy
from rclpy.node import Node
from soft_biometrics.inference import MTNSoftBiometrics, Collector
from demo_utils.soft_biometrics import BufferEmptyError
from messages.msg import *
import cv_bridge
from collections import deque
from demo_utils.ros_utils import *

class SoftBiometricsNode(Node):

    frequency: int

    PARAMETERS = [
        ROS2Param("frequency", value = 8, dest = "frequency"),
    ]

    NODE_NAME = "soft_biometrics_node"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAMETERS)
        self._queue = deque(maxlen=1)
        self._model = MTNSoftBiometrics(fp16=False)
        self._br = cv_bridge.CvBridge()
        self._collector = Collector()
        self.pub_info = self.create_publisher(SoftBioFrame, "/softbiometrics", 10)
        self.sub_image = self.create_subscription(FaceDetectorFrameData, "/asd/face_info", self._img_callback, 10)
        self._timer_read = self.create_timer(1.0 / self.frequency, self.node_callback)
        self.get_logger().info(f"{self.NODE_NAME} started")
    
    def node_callback(self, *args, **kwargs):
        msg_frame = SoftBioFrame()
        if len(self._queue) != 0:
            msg = self._queue.pop()
            faces = msg.faces
            faces_list = []
            info_list = []
            for i, face in enumerate(faces):
                img = self._br.compressed_imgmsg_to_cv2(face)
                faces_list.append(img)
            if len(faces_list) != 0:
                info_list = self._model(faces_list)
            assert len(msg.track_ids) == len(info_list)
            for info, track_id in zip(info_list, msg.track_ids):
                self._collector.add(track_id, info)
                try:
                    info = self._collector.get(track_id)
                except BufferEmptyError:
                    break
                info_msg = info.to_msg()
                info_msg.track_id = track_id
                msg_frame.info.append(info_msg)
        msg_frame.header.stamp = self.get_clock().now().to_msg()
        self.pub_info.publish(msg_frame)
    
    def _img_callback(self, msg: FaceDetectorFrameData):
        self._queue.append(msg)
    
    def loop(self):
        rclpy.spin(self)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = SoftBiometricsNode()
    node.loop()

if __name__ == "__main__":
    main()