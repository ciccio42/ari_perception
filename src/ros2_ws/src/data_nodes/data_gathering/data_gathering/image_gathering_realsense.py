#!/usr/bin/python3.8

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
import cv2 as cv
import cv_bridge
from messages.msg import MetaCamera, CompressedImageStamped
from messages.srv import MetaCameraSrv
import pyrealsense2 as rs
import numpy as np
from demo_utils.ros_utils import *

NODE_NAME = "image_input_node"
HFOV = 64
VFOV = 41

class Handler(Node):

    FPS: int
    WIDTH: int
    HEIGHT: int
    exposure: int

    PARAM_LIST = ROS2ParamList(
        ROS2Param(name="fps", dest="FPS", value=15),
        ROS2Param(name="width", dest="WIDTH", value=640),
        ROS2Param(name="height", dest="HEIGHT", value=480),
        ROS2Param(name="exposure", dest="exposure", value=60),
        )

    def __init__(self):
        super().__init__(NODE_NAME)
        self.logger = self.get_logger()
        init_parameters(self, self.PARAM_LIST)
        self.cap = self.open_stream()
        self._change_exposure(self.exposure)
        self.add_on_set_parameters_callback(self.set_exposure)
        self._is_configured = False
        self.br = cv_bridge.CvBridge()
        self.pub_img = self.create_publisher(CompressedImageStamped, "/img/compressed", qos_profile=10, callback_group=MutuallyExclusiveCallbackGroup())
        self.pub_meta = self.create_publisher(MetaCamera, "/meta/camera", qos_profile=LATCH_PROFILE)
        self.srv_meta = self.create_service(MetaCameraSrv, "/meta/camera", self._metacamera_callback)
        self.publish_meta()
        self._timer_read = self.create_timer(1e-4, self.read, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"{NODE_NAME} started")
        
    def _metacamera_callback(self, req: MetaCameraSrv.Request, resp: MetaCameraSrv.Response ):
        resp.fps = float(self.FPS)
        resp.width = self.WIDTH
        resp.height = self.HEIGHT
        resp.hfov = HFOV
        resp.vfov = VFOV
        return resp

    def set_exposure(self, params: List[Parameter]):
        param_target = None
        for p in params:
            if p.name == "exposure":
                param_target = p
        if param_target is None:
            return SetParametersResult(successful=False, reason="Exposure paramter is not in the input")
        actual_value = param_target.get_parameter_value().integer_value
        status  = self._change_exposure(actual_value)
        if not status:
            return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def _change_exposure(self, exposure: int):
        prev = self._get_exposure()
        if prev == exposure:
            self.logger.warn(f"Exposure: {exposure} is equal to the current one")
            return False
        sensor_rgb = self.cap.get_active_profile().get_device().query_sensors()[1]
        try:
            sensor_rgb.set_option(rs.option.exposure, exposure)
            self.logger.warn(f"Changing exposure to: {exposure}")
            return True
        except:
            self.logger.error("Exposure param no changed")
            return False
    
    def _get_exposure(self):
        sensor_rgb = self.cap.get_active_profile().get_device().query_sensors()[1]
        exposure = sensor_rgb.get_option(rs.option.exposure)
        return exposure

    def publish_meta(self):
        
        msg = MetaCamera(
            fps=float(self.FPS),
            width=self.WIDTH,
            height=self.HEIGHT,
            hfov=HFOV,
            vfov=VFOV
        )
        self.pub_meta.publish(msg)
    
    def open_stream(self):
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.WIDTH, self.HEIGHT, rs.format.bgr8, int(self.FPS))
        pipe.start(config)
        return pipe
    
    def read(self):
        frames = self.cap.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_frame_np = np.asanyarray(color_frame.get_data())
        msg = CompressedImageStamped()
        img_msg = self.br.cv2_to_compressed_imgmsg(color_frame_np)
        msg.image = img_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_img.publish(msg)
    
    def loop(self):
        rclpy.spin(self)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = Handler()
    EXECUTOR = MultiThreadedExecutor(num_threads=2)
    node.loop()

if __name__ == "__main__":
    main()
    