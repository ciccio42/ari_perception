import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from object_detector.inference import YOLOv8TensorRT
from collections import deque
import cv2 as cv
import cv_bridge
import numpy as np
from messages.msg import *
from demo_utils.ros_utils import *

class ObjDetectorNode(Node):

    NODE_NAME = "obj_detector_node"

    frequency: int
    confidence_threshold: float
    
    PARAMETERS = [
        ROS2Param("frequency", value = 7, dest = "frequency"),
        ROS2Param("confidence_threshold", value = 0.7, dest ="confidence_threshold")
    ]

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAMETERS)
        self._br = cv_bridge.CvBridge()
        self._buffer = deque(maxlen=1)
        self.sub_img = self.create_subscription(CompressedImageStamped, "/img/compressed", self.image_callback, 1)
        self._pub_objects = self.create_publisher(ObjectFrameStamped, "/obj", 10)
        self._model = YOLOv8TensorRT()
        self._timer = self.create_timer(1.0 / self.frequency, self._process)
        self.get_logger().info(f"{self.NODE_NAME} started")

    def loop(self):
        rclpy.spin(self)
    
    def image_callback(self, msg):
        self._buffer.append(msg)

    def _process(self, *args, **kwargs):
        if len(self._buffer) != 0:
            out_msg = ObjectFrameStamped()
            msg: CompressedImageStamped = self._buffer.pop()
            img = self._br.compressed_imgmsg_to_cv2(msg.image)
            classes, confidences, boxes, names = self._model(img)
            # print("names:", names, "confidences", confidences)
            for name, confidence, box in zip(names, confidences, boxes):
                if confidence < self.confidence_threshold:
                    continue
                box = np.array(box, dtype=np.uint32).flatten()
                msg = Object(object=name, confidence=float(confidence), boxes=box)
                out_msg.objects.append(msg)
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self._pub_objects.publish(out_msg)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = ObjDetectorNode()
    node.loop()

if __name__ == "__main__":
    main()