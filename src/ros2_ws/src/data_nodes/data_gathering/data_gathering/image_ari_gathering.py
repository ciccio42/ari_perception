# #!/usr/bin/python3

# import rclpy
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
# import math
# import cv2
# import numpy as np
# from sensor_msgs.msg import CompressedImage, CameraInfo
# from messages.msg import MetaCamera, CompressedImageStamped
# from messages.srv import MetaCameraSrv

# NODE_NAME = "image_ari_input_node"

# class AriCameraHandler(Node):
#     def __init__(self):
#         super().__init__(NODE_NAME)
#         self.logger = self.get_logger()

#         self.declare_parameter("fps", 15.0)
#         self.declare_parameter("width", 640)
#         self.declare_parameter("height", 480)
        
#         self.hfov = 64.0
#         self.vfov = 41.0
#         self._info_received = False

#         self.cb_group = MutuallyExclusiveCallbackGroup()

#         latch_qos = QoSProfile(
#             depth=1,
#             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
#             history=QoSHistoryPolicy.KEEP_LAST
#         )

#         self.pub_img = self.create_publisher(
#             CompressedImageStamped, 
#             "/img/compressed", 
#             qos_profile=10, 
#             callback_group=self.cb_group
#         )
        
#         self.pub_meta = self.create_publisher(
#             MetaCamera, 
#             "/meta/camera", 
#             qos_profile=latch_qos
#         )

#         self.srv_meta = self.create_service(
#             MetaCameraSrv, 
#             "/meta/camera", 
#             self._metacamera_callback,
#             callback_group=self.cb_group
#         )

#         self.sub_img = self.create_subscription(
#             CompressedImage,
#             "/head_front_camera/color/image_raw/compressed", #"/head_front_camera/image_throttle/compressed", #"/head_front_camera/color/image_raw/compressed", 
#             self._image_bridge_callback,
#             1,
#             callback_group=self.cb_group
#         )
        
#         self.sub_info = self.create_subscription(
#             CameraInfo,
#             "/head_front_camera/color/camera_info", 
#             self._camera_info_bridge_callback,
#             10,
#             callback_group=self.cb_group
#         )

#         self.publish_meta()
        
#         self.logger.info(f"{NODE_NAME} started.")
#         self.logger.info("Resizing and encoding will be performed locally in ROS 2.")

#     def _camera_info_bridge_callback(self, msg: CameraInfo):
#         """Calculates FOV from standard CameraInfo intrinsics."""
#         if self._info_received:
#             return

#         fx = msg.k[0]
#         fy = msg.k[4]
#         width = msg.width
#         height = msg.height

#         if fx > 0 and fy > 0:
#             self.hfov = math.degrees(2 * math.atan(width / (2.0 * fx)))
#             self.vfov = math.degrees(2 * math.atan(height / (2.0 * fy)))
            
#             self._info_received = True
#             self.logger.info(f"Intrinsics updated! HFOV: {self.hfov:.2f}°, VFOV: {self.vfov:.2f}°")
#             self.publish_meta()
#         else:
#             self.logger.warn("Received invalid focal lengths. Keeping defaults.")

#     def _get_current_params(self):
#         """Helper to get current manually set parameters."""
#         return {
#             'fps': self.get_parameter("fps").value,
#             'width': self.get_parameter("width").value,
#             'height': self.get_parameter("height").value,
#         }

#     def _metacamera_callback(self, req: MetaCameraSrv.Request, resp: MetaCameraSrv.Response):
#         """Service callback to return camera metadata."""
#         params = self._get_current_params()
        
#         resp.fps = float(params['fps'])
#         resp.width = int(params['width'])
#         resp.height = int(params['height'])
#         resp.hfov = int(self.hfov) 
#         resp.vfov = int(self.vfov) 
        
#         return resp

#     def publish_meta(self):
#         """Publishes the latched MetaCamera message."""
#         params = self._get_current_params()
        
#         msg = MetaCamera(
#             fps=float(params['fps']),
#             width=int(params['width']),
#             height=int(params['height']),
#             hfov=int(self.hfov),
#             vfov=int(self.vfov) 
#         )
#         self.pub_meta.publish(msg)

#     def _image_bridge_callback(self, incoming_msg: CompressedImage):
#         """Decodes, resizes, re-encodes, and wraps the image into the custom message."""
#         self.get_logger().info("Receiving image")
#         try:
#             np_arr = np.frombuffer(incoming_msg.data, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
#             target_width = self.get_parameter("width").value
#             target_height = self.get_parameter("height").value
#             resized_image = cv2.resize(cv_image, (target_width, target_height))
            
#             # flip the image
#             # cv2.imwrite('pre_flip.jpg', resized_image)
#             resized_image = cv2.flip(resized_image, 1)
#             # cv2.imwrite('post_flip.jpg', resized_image)
            
#             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
#             _, encoded_image = cv2.imencode('.jpg', resized_image, encode_param)
            
#             inner_image = CompressedImage()
#             inner_image.header = incoming_msg.header 
#             inner_image.format = "jpeg"
#             inner_image.data = np.array(encoded_image).tobytes()

#             msg_out = CompressedImageStamped()
#             msg_out.header.stamp = self.get_clock().now().to_msg()
#             msg_out.image = inner_image
            
#             self.pub_img.publish(msg_out)
            
#         except Exception as e:
#             self.logger.error(f"Failed to process and wrap image: {e}")

# def main():
#     if not rclpy.ok():
#         rclpy.init()
        
#     node = AriCameraHandler()
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
    
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         node.logger.info("Shutting down node...")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()

#!/usr/bin/python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, ReliabilityPolicy
import math
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, CameraInfo
from messages.msg import MetaCamera, CompressedImageStamped
from messages.srv import MetaCameraSrv
import threading

NODE_NAME = "image_ari_input_node"

# Pre-build the JPEG encode param once — not per frame
ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

class AriCameraHandler(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.logger = self.get_logger()

        self.declare_parameter("fps", 15.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        self.hfov = 64.0
        self.vfov = 41.0
        self._info_received = False

        # Cache target dimensions — avoids get_parameter() on every frame
        self._target_width = self.get_parameter("width").value
        self._target_height = self.get_parameter("height").value

        # Lock for thread-safe dimension updates
        self._dim_lock = threading.Lock()

        # Separate callback groups:
        # - image processing gets its own group (can be reentrant if you increase threads)
        # - service/meta gets its own so it's never blocked by image work
        self._img_cb_group  = MutuallyExclusiveCallbackGroup()
        self._meta_cb_group = MutuallyExclusiveCallbackGroup()

        latch_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Use BEST_EFFORT for image topic — drops stale frames instead of queuing them
        img_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT  # don't queue stale frames
        )

        self.pub_img = self.create_publisher(
            CompressedImageStamped,
            "/img/compressed",
            qos_profile=10,
            callback_group=self._img_cb_group
        )

        self.pub_meta = self.create_publisher(
            MetaCamera,
            "/meta/camera",
            qos_profile=latch_qos
        )

        self.srv_meta = self.create_service(
            MetaCameraSrv,
            "/meta/camera",
            self._metacamera_callback,
            callback_group=self._meta_cb_group  # isolated — never blocked by image work
        )

        self.sub_img = self.create_subscription(
            CompressedImage,
            "/head_front_camera/color/image_raw/compressed",
            #"/head_front_camera/image_throttle/compressed",
            #"/head_front_camera/color/image_raw/compressed",
            self._image_bridge_callback,
            img_qos,
            callback_group=self._img_cb_group
        )

        self.sub_info = self.create_subscription(
            CameraInfo,
            "/head_front_camera/color/camera_info",
            self._camera_info_bridge_callback,
            10,
            callback_group=self._meta_cb_group  # lightweight, share with meta
        )

        self.publish_meta()
        self.logger.info(f"{NODE_NAME} started.")

    def _camera_info_bridge_callback(self, msg: CameraInfo):
        if self._info_received:
            return

        fx, fy = msg.k[0], msg.k[4]
        if fx > 0 and fy > 0:
            self.hfov = math.degrees(2 * math.atan(msg.width  / (2.0 * fx)))
            self.vfov = math.degrees(2 * math.atan(msg.height / (2.0 * fy)))
            self._info_received = True
            self.logger.info(f"Intrinsics updated! HFOV: {self.hfov:.2f}°, VFOV: {self.vfov:.2f}°")
            self.publish_meta()
        else:
            self.logger.warn("Received invalid focal lengths. Keeping defaults.")

    def _get_current_params(self):
        return {
            'fps':    self.get_parameter("fps").value,
            'width':  self.get_parameter("width").value,
            'height': self.get_parameter("height").value,
        }

    def _metacamera_callback(self, req, resp):
        params = self._get_current_params()
        resp.fps    = float(params['fps'])
        resp.width  = int(params['width'])
        resp.height = int(params['height'])
        resp.hfov   = int(self.hfov)
        resp.vfov   = int(self.vfov)

        # Sync cache if someone changed params externally
        with self._dim_lock:
            self._target_width  = resp.width
            self._target_height = resp.height

        return resp

    def publish_meta(self):
        params = self._get_current_params()
        msg = MetaCamera(
            fps=float(params['fps']),
            width=int(params['width']),
            height=int(params['height']),
            hfov=int(self.hfov),
            vfov=int(self.vfov)
        )
        self.pub_meta.publish(msg)

    def _image_bridge_callback(self, incoming_msg: CompressedImage):
        try:
            # np.frombuffer avoids a copy vs. np.array(bytes)
            np_arr = np.frombuffer(incoming_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.logger.warn("imdecode returned None — skipping frame")
                return

            with self._dim_lock:
                w, h = self._target_width, self._target_height

            # Combine resize + flip into one pass using flip-aware dst
            # cv2.resize then cv2.flip is the minimum; they can't fuse further in CPU OpenCV
            resized = cv2.resize(cv_image, (w, h), interpolation=cv2.INTER_LINEAR)
            flipped = cv2.flip(resized, 1)

            # imencode returns a contiguous buffer — use .tobytes() directly (zero extra copy)
            success, encoded = cv2.imencode('.jpg', flipped, ENCODE_PARAMS)
            if not success:
                self.logger.warn("imencode failed — skipping frame")
                return

            inner = CompressedImage()
            inner.header = incoming_msg.header
            inner.format = "jpeg"
            inner.data   = encoded.tobytes()  # single copy, no np.array() wrapper

            msg_out = CompressedImageStamped()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.image = inner

            self.pub_img.publish(msg_out)

        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")


def main():
    if not rclpy.ok():
        rclpy.init()

    node = AriCameraHandler()

    # 4 threads: image decode/encode is the heavy hitter, give it room
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.logger.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()