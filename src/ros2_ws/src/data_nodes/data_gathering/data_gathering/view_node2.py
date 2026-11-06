import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
from messages.msg import *
import cv2 as cv
import cv_bridge
from typing import List
import numpy as np
from collections import deque, OrderedDict
from demo_utils.soft_biometrics import SoftBiometricsInfo
from demo_utils.ros_utils import *
from PIL import Image

NODE_NAME = "view_node"
FONT = cv.FONT_HERSHEY_SIMPLEX
SCALE = 0.7
THICKNESS = 2

class Handler(Node):

    PARAMS = ROS2ParamList(
        ROS2Param("show_bio", dest="_show_biometrics", value=True, description="Flag if show biometrics on Web GUI"),
        ROS2Param("show_asd", dest="_show_asd_info", value=False, description="Flag if show ASD data"),
        ROS2Param("show_obj", dest="_show_obj", value=True, description="Flag if show detected objects"),
    )

    def __init__(self):
        super().__init__(NODE_NAME)
        init_parameters(self, self.PARAMS)
        self.add_on_set_parameters_callback(self.dynamic_reconfigure)
        self.logger = self.get_logger()
        self._br = cv_bridge.CvBridge()
        self._pub_view_node = self.create_publisher(
            CompressedImage,
            "/view/image/compressed",
            10
        )

        self._sub_is_looking = self.create_subscription(
            ModelsSyncDataStamped,
            "/models_data_sync/data",
            self._data_sync_callback,
            1
        )

        self.logger.info("View node started")
    
    def dynamic_reconfigure(self, params):
        return default_reconfigure(self, params, self.PARAMS)

    def _data_sync_callback(self, msg: ModelsSyncDataStamped):
        frame = self._msg2img(msg.frame)
        pil_img = Image.fromarray(frame[:,:,::-1])
        # pil_img.save('clean_img.jpg')
        boxes_all = self._convert2array(msg.face_info.boxes, dtype=np.uint32)
        track_ids_face = list(msg.face_info.track_ids)

        is_looking_list, yaw_angles = self._extract_gaze_data(msg.is_looking_data)
        is_speaking_list, track_ids_asd = self._extract_asd_data(msg.is_speaking_data)

        color_list_all, text_list_all = self._color_selector(
            is_looking_list,
            is_speaking_list,
            track_ids_face,
            yaw_angles
        )

        boxes = boxes_all
        color_list = color_list_all
        text_list = text_list_all
        soft_biometrics_text = self._extract_soft_biometrics(msg.biometrics)
        
        frame = self._draw_objects(frame, self._show_obj, msg.objects)
        # print(f"Soft biometrics text {soft_biometrics_text}")
        out_img = self._draw_boxes(frame, 
                                   self._show_asd_info, self._show_biometrics,
                                   boxes, color_list, text_list, soft_biometrics_text)
        
        out_img = cv.resize(out_img, (480, 640))
        # save image with PIL
        pil_img = Image.fromarray(out_img[:,:,::-1])
        # pil_img.save('tablet_view.jpg')

        out_msg = self._img2msg(out_img)
        self._pub_view_node.publish(out_msg)
        self.logger.info("View node receiving", once=True)
    
    def _extract_soft_biometrics(self, bio_frame: SoftBioFrame):
        biometrics_text = []
        for info in bio_frame.info:
            track_id = info.track_id
            bio = SoftBiometricsInfo.from_msg(info)
            bio: SoftBiometricsInfo
            age = round(bio.age, 0)
            text = f"ID: {track_id}\n{bio.gender.value}, Age: {age:.0f}\n{bio.emotion.value}, {bio.ethnicity.value}"
            biometrics_text.append(text)
        return biometrics_text

    def _filter_tracks(self, master_tracks, data_and_tracks):
        nw_data = []
        nw_tracks = []
        data_list = data_and_tracks[0]
        tracks_list = data_and_tracks[1]
        assert len(tracks_list) == len(data_list)
        for track, data in zip(tracks_list, data_list):
            if track in master_tracks:
                nw_data.append(data)
                nw_tracks.append(track)
        return nw_tracks, nw_data
    
    def _msg2img(self, img_msg) -> CompressedImage:
        return self._br.compressed_imgmsg_to_cv2(img_msg)
    
    def _img2msg(self, img: CompressedImage):
        return self._br.cv2_to_compressed_imgmsg(img)
    
    def _convert2array(self, msg, dtype=np.float32) -> np.ndarray:
        dim = msg.layout.dim
        size = [s.size for s in dim]
        data = msg.data
        arr = np.array(data, dtype=dtype)
        arr = np.reshape(arr, size)
        return arr
    
    def _draw_objects(self, img, show_obj, objs_msg: ObjectFrameStamped):
        if not show_obj:
            return img
        objects = objs_msg.objects
        for obj in objects:
            name: str = obj.object
            box: list = obj.boxes
            img = cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(255,255,0), thickness=THICKNESS)
            img = cv.putText(img, name, (box[0], box[1]-10), cv.FONT_ITALIC, fontScale=0.65, color=(0,0,0), thickness=THICKNESS)
        return img

    def _draw_boxes(self, img, 
                    show_asd_info, show_biometrics,
                    boxes, color_list: list, text_list: list, soft_biometrics_text_list: list):
        def _check_boxes(boxes, color_list):
            dim_color_list = len(color_list)
            dim_boxes = len(boxes)
            if len(boxes) > len(color_list):
                boxes = boxes[:dim_color_list]
            elif len(boxes) < len(color_list):
                color_list = color_list[:dim_boxes]
            return boxes, color_list
        
        if len(boxes) == 0:
            return img
        
        boxes, color_list = _check_boxes(boxes, color_list)
        assert len(boxes) == len(color_list), f"{boxes} - {color_list}"
        for box, color, text, soft_text  in zip(boxes, color_list, text_list, soft_biometrics_text_list):
            text_split = text.split("\n")
            soft_split = soft_text.split("\n")
            box_height = box[3] - box[1]
            box_width = box[2] - box[0]
            try:
                img = cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            except TypeError:
                continue

            if show_asd_info:
                for i, t in enumerate(text_split):
                    text_orig = (int(box[0]), int(box[1]-((i+1)*-15)))
                    img = cv.putText(img, t, text_orig, fontFace=cv.FONT_ITALIC, fontScale=0.5, color=color, thickness=1)
            
            if show_biometrics:
                # img = cv.rectangle(img,  white_box_orig[0], white_box_orig[1], (255, 255, 255), -1)
                size_list = []
                offset = 20
                for i, t in enumerate(soft_split):
                    size_list.append(cv.getTextSize(t, fontFace=FONT, fontScale=SCALE, thickness=THICKNESS)[0])
                
                text_width = int(max([x[0] for x in size_list]))
                text_height = int(max([x[1] for x in size_list]))
                orig_upper_left = box[0] - box_width//2
                white_box_orig = ((orig_upper_left, box[1] + box_height+5), (orig_upper_left +text_width, box[3]+text_height*4))
                img = cv.rectangle(img, white_box_orig[0], white_box_orig[1], (255,255,255), -1)

                for i, t in enumerate(soft_split):
                    text_orig = (orig_upper_left, box[1]+box_height+(i+1)*offset)
                    img = cv.putText(img, t, text_orig, fontFace=FONT, fontScale=SCALE, color=(0,0,0), thickness=THICKNESS)

        return img
    
    def _extract_gaze_data(self, msg: GazeData):
        assert isinstance(msg, GazeData)
        gaze_data_list: List[GazeFrameData] = msg.gaze_data
        is_looking_list = []
        yaw_angles = []
        for e in gaze_data_list:
            is_looking_list.append(e.is_looking)
            yaw_angles.append(e.yaw)
        return is_looking_list, yaw_angles
    
    def _extract_asd_data(self, msg: AsdData):
        assert isinstance(msg, AsdData)
        asd_data_list: List[AsdFrameData] = msg.asd_data
        asd_data_res_list = []
        track_ids = []
        for e in asd_data_list:
            asd_data_res_list.append(e.asd)
            track_ids.append(e.track_id)
        return asd_data_res_list, track_ids
    
    def _color_selector(self, is_looking_list, asd_data_list, track_ids, yaw_angles):
        color_list = []
        text_list = []
        for i, track_id in enumerate(track_ids):
            
            try:
                is_looking = is_looking_list[i]
                asd_data = asd_data_list[i]
                yaw = yaw_angles[i]
                yaw = round(yaw, 2)
            except IndexError:
                is_looking = None
                asd_data = None
                yaw = None
                
            #Color in BGR
            if is_looking == 1 and asd_data == 0:
                color = (0,192,255) #Yellow
                text = f"looking\nYaw: {yaw}\nNO speaking"
            elif is_looking == 0 and asd_data == 1:
                color = (255,0,0) #Blue
                text = f"NO looking\nYaw: {yaw}\nspeaking"
            elif is_looking == 1 and asd_data == 1:
                color = (0,255,0) #Green
                text = f"looking\nYaw: {yaw}\nspeaking"
            elif is_looking == 0 and asd_data == 0:
                color = (0,0,255) #Red
                text = f"NO looking\nYaw: {yaw}\nNO speaking"
            else:
                color = 255,150,150 #violet
                text = ""
                
            text += f"\ntrack: {track_id}"
            color_list.append(color)
            text_list.append(text)
        return color_list, text_list
    
    def _check_for_data(self, buffer):
        return len(buffer) > 0
    
    def loop(self):
        rclpy.spin(self)

def main():
    if not rclpy.ok():
        rclpy.init()
    node = Handler()
    node.loop()

if __name__ == "__main__":
    main()