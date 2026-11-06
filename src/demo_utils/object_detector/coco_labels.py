import yaml
from pathlib import Path

class COCO:

    _labels: dict

    def __init__(self):
        curr_dir = Path(__file__).parent
        labels_path = curr_dir.joinpath("coco8.yaml")
        with open(str(labels_path)) as fil:
            self._labels = yaml.safe_load(fil)["names"]
    
    @property
    def labels(self):
        return self._labels