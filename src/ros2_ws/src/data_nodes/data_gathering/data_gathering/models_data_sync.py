import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from messages.msg import *
from demo_utils.ros_utils import *
import message_filters

NODE_NAME = "model_data_sync_node"

class Handler(Node):

    slop: float

    PARAM_LIST = [
        ROS2Param(name="slop", value=0.2, dest="slop"),
    ]

    def __init__(self):
        super().__init__(NODE_NAME)
        init_parameters(self, self.PARAM_LIST)
        self.pub_models_data_sync = self.create_publisher(ModelsSyncDataStamped, "/models_data_sync/data", qos_profile=10)
        sub_topics = self.register_callbacks()
        syncronizer = message_filters.ApproximateTimeSynchronizer(sub_topics, 1, self.slop, allow_headerless=True)
        syncronizer.registerCallback(self.callback)
        self.get_logger().info("Models data syncronizer started")

    def register_callbacks(self):
        sub_is_looking = message_filters.Subscriber(self, GazeData, "/gaze/is_looking")
        sub_active_speaksing = message_filters.Subscriber(self, AsdData, "/asd/speaking")
        sub_frame = message_filters.Subscriber(self, CompressedImageStamped, "/img/compressed",)
        sub_asd_info = message_filters.Subscriber(self, FaceDetectorFrameData, "/asd/face_info")
        sub_biometrics = message_filters.Subscriber(self, SoftBioFrame, "/softbiometrics")
        sub_objs = message_filters.Subscriber(self, ObjectFrameStamped, "/obj")
        return [sub_is_looking, sub_active_speaksing, sub_frame, sub_asd_info, sub_biometrics, sub_objs]

    def callback(self, is_looking_msg: GazeData, 
                 is_speaking_data_msg: AsdData, 
                 frame: CompressedImageStamped,
                 face_info: FaceDetectorFrameData,
                 biometrics: SoftBioFrame,
                 objs: ObjectFrameStamped):
        header = Header(stamp=self.get_clock().now().to_msg())
        msg = ModelsSyncDataStamped(header=header, 
                                    frame=frame.image,
                                    is_looking_data=is_looking_msg, 
                                    is_speaking_data=is_speaking_data_msg,
                                    face_info=face_info,
                                    biometrics=biometrics,
                                    objects=objs)
        self.get_logger().debug("Publishing")
        self.pub_models_data_sync.publish(msg)
    
    def loop(self):
        rclpy.spin(self)
    
def main():
    if not rclpy.ok():
        rclpy.init()
    node = Handler()
    node.loop()

if __name__ == "__main__":
    main()