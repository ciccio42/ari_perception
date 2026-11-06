import rclpy.exceptions
from sensor_msgs.msg import CompressedImage
from messages.msg import CompressedImageStamped
from std_srvs.srv import SetBool

import cv2
import numpy as np
import rclpy 
from demo_utils.hand_detector.hand_tracking import HandDetector
from cv_bridge import CvBridge
import imutils
from messages.srv import MoveConnect4Srv
from std_msgs.msg import String
from collections import deque
from demo_utils.ros_utils import *
import logging


class HandConnectFourCallback():
    """
        Class to process hand detection data for Connect Four game moves.

        Attributes:
            _buffer (deque): Buffer to store the latest image messages.
            hand_publisher (Publisher): Publishes processed hand images.
            move_service (Service): Service to get the recognized move.
            pub_move (Publisher): Publishes detected moves as strings.
            detector (HandDetector): Detect hands and count the number of fingers up.
            bridge (CvBridge): Converts between ROS and OpenCV image formats.
            current_move (str): Stores the currently recognized move.
            rtv_size (tuple): Resolution size for image processing.
            prev_move (int): Tracks the previous move for consistency checks.
            move_count (int): Counter for consistent frame detection.
            required_consistent_frames (int): Frames required to confirm a move.
            hands_present_prev_frame (bool): Tracks hand presence in the previous frame.
            move_sent (bool): Flag indicating if a move has been sent.
        """
    
    
    def __init__(self, node: Node):
        self.node = node
        self._buffer = deque(maxlen=1)
        self.hand_publisher = self.node.create_publisher(CompressedImage, "/view/connect_four_game/hand_rgb/compressed", 1)
        self.move_service = self.node.create_service(MoveConnect4Srv, '/game/connect_four/move', self.get_recognized_move)
        self.pub_move = self.node.create_publisher(String, "/game/connect_four/move", 10)
        self.detector = HandDetector(maxHands=2)
        self.bridge = CvBridge()
        self.current_move  = - 1
        self.rtv_size = (320, 250)
        self.node.create_timer(1.0 / self.node.frequency, self.callback)

        self.prev_move = None
        self.move_count = 0
        self.required_consistent_frames = 5
        self.hands_present_prev_frame = False  
        self.move_sent = False 


    def get_recognized_move(self, req): 
        """
        ROS service callback to return the current recognized move.
        
        Parameters:
            req (MoveConnect4SrvRequest): Service request object.
        Returns:
            int: The currently recognized move.
        """
        return self.current_move
    

    def __call__(self, msg_img: CompressedImageStamped):
        """
        Adds a new image message to the buffer.
        
        Parameters:
            msg_img (CompressedImageStamped): Incoming image message.
        """
        self._buffer.append(msg_img)

    def callback(self, *args, **kwargs):
        """
        Processes buffered images for hand detection and move recognition.
        Detects hands in the image.
        Counts fingers and determines moves.
        Publishes recognized moves if consistent for a set number of frames.
        """
        if len(self._buffer) != 0:
            msg_img: CompressedImageStamped = self._buffer.pop()
            img = msg_img.image
            img = self.bridge.compressed_imgmsg_to_cv2(img)

            img = imutils.resize(img, height=self.rtv_size[0])
            wc = int(img.shape[1] / 2)
            l = int(wc - self.rtv_size[0] / 2)
            r = int(wc + self.rtv_size[1] / 2)
            img = img[:, l:r]
            img = cv2.flip(img, 1)

            hands, img, bbox = self.detector.findHands(img)
            recognizedMove = -1

            if hands:
                total_fingers = 0 

                for hand in hands:
                    fingers = self.detector.count_fingers_with_angles(hand)
                    total_fingers += fingers

                if 1 <= total_fingers <= 7:
                    recognizedMove = total_fingers
                    self.current_move = str(recognizedMove)
                else:
                    self.current_move = "InvalidMove"
            else:
                recognizedMove = -1
                self.current_move = "NoHand"

            if recognizedMove == self.prev_move:
                self.move_count += 1
            else:
                self.move_count = 0
                self.prev_move = recognizedMove

            if recognizedMove != -1 and not self.move_sent and self.hands_present_prev_frame is False:
                cv2.putText(img, self.current_move, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                if self.move_count >= self.required_consistent_frames:

                    if recognizedMove >= 1 and recognizedMove <= 7:
                        move = recognizedMove - 1
                    else:
                        move = -1
                    msg = String()
                    msg.data = str(move)
                    self.pub_move.publish(msg)

                    self.move_count = 0
                    self.move_sent = True 
                    self.hands_present_prev_frame = True 
            
            msg_img = self.bridge.cv2_to_compressed_imgmsg(img)
            msg_img.header.stamp = self.node.get_clock().now().to_msg()
            self.hand_publisher.publish(msg_img)


            if recognizedMove == -1:
                self.move_sent = False  
                self.hands_present_prev_frame = False  

class Connect4HandLandmarkNode(Node):

    frequency: int 

    PARAMETERS = [
        ROS2Param("frequency", value=7, dest ="frequency")
    ]

    NODE_NAME = "image_hand_fingers_node"

    """
    ROS Node for handling hand landmark detection in a Connect Four game.

    This node subscribes to a compressed image topic and processes the input using a callback.
    The subscription can be activated or deactivated dynamically through a service.
    """

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAMETERS)
        self._active = False #TODO service
        
        # Initialize the image processing callback
        self.image_callback = HandConnectFourCallback(self)

        # Service to toggle active state of the node
        self._srv_active = self.create_service(SetBool, "/game/connect_four/active", self._active_callback)
        self.get_logger().info("Connection four node created.")

    def loop(self):
        """
        Keeps the ROS node alive by spinning indefinitely.
        """
        rclpy.spin(self)

    def create_sub(self):
        """
        Creates the image subscriber.

        Subscribes to the '/img/compressed' topic with a callback for processing images.
        """
        self.sub_img = self.create_subscription(
            CompressedImageStamped, "/img/compressed", self.image_callback, qos_profile=1
        )
        self.get_logger().info("Image subscriber created.")

    def destroy_sub(self):
        """
        Destroys the image subscriber.

        Unregisters the subscriber and deletes the reference.
        """
        #self.sub_img.unregister()
        #del self.sub_img
        self.destroy_subscription(self.sub_img)
        self.get_logger().info("Image subscriber destroyed.")

    def _active_callback(self, request: SetBool.Request, response: SetBool.Response):
        self._active = request.data

        if self._active and not hasattr(self, "sub_img"):
            self.create_sub()
        elif not self._active and hasattr(self, "sub_img"):
            self.destroy_sub()
        else:
            logging.warning("Operation not permitted")

        response.success = True
        return response
    

def main():
    try:
        if not rclpy.ok():
            rclpy.init()
        landamrk = Connect4HandLandmarkNode()
        landamrk.loop()
    except rclpy.exceptions.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()