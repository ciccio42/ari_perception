import numpy as np
from asd.sort_tracker import Sort


class TrackerSort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.05):
        self._tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold) 
        self._detection_class = 0
    
    def _update_tracks(self, boxes, probs):
        raw_detections = []
        if len(boxes) == 0:
             tracks = self._tracker.update()
             return tracks
        for box, p in zip(boxes, probs):
                
            if isinstance(box, np.ndarray):
                box = box.flatten().tolist()
            
            if isinstance(p, np.ndarray):
                p = float(p.item())
            
            assert len(box) == 4
            
            raw_detection = [box[0], box[1], box[2], box[3], p, self._detection_class]
            raw_detections.append(raw_detection)
        raw_detections = np.array(raw_detections)
        tracks = self._tracker.update(raw_detections)
        return tracks
