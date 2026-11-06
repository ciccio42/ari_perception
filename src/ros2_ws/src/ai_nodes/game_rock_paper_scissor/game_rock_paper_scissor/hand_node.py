from sensor_msgs.msg import CompressedImage
from messages.msg import *
from std_srvs.srv import SetBool
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from demo_utils.hand_detector.hand_tracking import HandDetector
from cv_bridge import CvBridge
import imutils
from messages.srv import MoveSrv
from collections import deque
from demo_utils.ros_utils import *
import rclpy.exceptions

class HandCallback:

    def __init__(self, node: Node):
        self.node = node
        self._buffer = deque(maxlen=1)
        self.hand_publisher = self.node.create_publisher(CompressedImage, "/view/game/hand_rgb/compressed", 1)
        self.move_service = self.node.create_service(MoveSrv, '/game/rock_paper_scissor/move', self.get_recognized_move)
        self.pub_move = self.node.create_publisher(Move, "/game/rock_paper_scissor/move", 10)
        self.detector = HandDetector(maxHands=1)
        self.bridge = CvBridge()
        self.move = [Move.ROCK, Move.PAPER, Move.SCISSOR, '']
        self.current_move  = ""
        self.rtv_size = (320,250)
        self.node.create_timer(1.0 / self.node.frequency, self.callback)

    def get_recognized_move(self, req: MoveSrv.Request, resp: MoveSrv.Response): 
        resp.move = self.current_move
        return resp

    def __call__(self, msg_img: CompressedImageStamped):
        self._buffer.append(msg_img)

    def callback(self, *args, **kwargs):
        if len(self._buffer) != 0:
            msg_img: CompressedImageStamped = self._buffer.pop()
            img = msg_img.image
            img = self.bridge.compressed_imgmsg_to_cv2(img)

            img = imutils.resize(img, height=self.rtv_size[0])
            wc = int(img.shape[1]/2)
            l= int(wc-self.rtv_size[0]/2)
            r= int(wc+self.rtv_size[1]/2)
            img = img[:,l:r]

            hands, img, bbox = self.detector.findHands(img)
            recognizedMove = 4

            if hands:
                recognizedMove = 4
                hand = hands[0]
                fingers = self.detector.fingersUp(hand)
                if fingers[1:] == [0, 0, 0, 0]:
                    recognizedMove = 1 #Rock
                if fingers[1:] == [1, 1, 1, 1]:
                    recognizedMove = 2 #Paper
                if fingers[1:] == [1, 1, 0, 0]:
                    recognizedMove = 3 #Scissor

                self.current_move = self.move[recognizedMove-1]
            else:
                recognizedMove = -1
                self.current_move = Move.NO_HAND

            if recognizedMove != -1: 
                cv2.putText(img, self.move[recognizedMove-1], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2) 

            msg_img = self.bridge.cv2_to_compressed_imgmsg(img)    
            msg_img.header.stamp = self.node.get_clock().now().to_msg()
            self.hand_publisher.publish(msg_img)
            if recognizedMove == 1:
                move = Move.ROCK
            elif recognizedMove == 2:
                move = Move.PAPER
            elif recognizedMove == 3:
                move = Move.SCISSOR
            else:
                move = Move.NO_HAND
            self.current_move = move
            self.pub_move.publish(Move(move=move))

class HandLandmarkNode(Node):

    NODE_NAME = "image_hand_node"

    frequency: int

    PARAMETERS = [
        ROS2Param("frequency", value = 8, dest = "frequency"),
    ]

    def __init__(self):
        super().__init__(self.NODE_NAME)
        init_parameters(self, self.PARAMETERS)
        self._active = False #TODO service
        self.image_callback = HandCallback(self)
        self._srv_active = self.create_service(SetBool, "/game/rock_paper_scissor/active", self._active_callback)


    def loop(self):
        rclpy.spin(self)

    def create_sub(self):
        self.sub_img = self.create_subscription(CompressedImageStamped, "/img/compressed", self.image_callback, qos_profile=1)
    
    def destroy_sub(self):
        #self.sub_img.unregister()
        self.destroy_subscription(self.sub_img)
        #del self.sub_img
    
    def _active_callback(self, request: SetBool.Request, response: SetBool.Response):
        self._active = request.data
        if self._active and not hasattr(self, "sub_img"):
            self.create_sub()
        elif not self._active and hasattr(self, "sub_img"):
            self.destroy_sub()
        else:
            self.get_logger().warn("Operation not permitted")
        
        response.success = True
        return response

def main():
    try:
        if not rclpy.ok():
            rclpy.init()
        landmark = HandLandmarkNode()
        landmark.loop()
    except rclpy.exceptions.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()