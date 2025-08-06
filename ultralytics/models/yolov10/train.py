from ultralytics.models.yolo.detect import DetectionTrainer
from .val import YOLOv10DetectionValidator
from .model import YOLOv10DetectionModel
from copy import copy
from ultralytics.utils import RANK
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'


class YOLOv10DetectionTrainer(DetectionTrainer):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo", 
        return YOLOv10DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = YOLOv10DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


