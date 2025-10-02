import os
from typing import Any, Optional

from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image as PILImage

def _ensure_bgr(img: Any):
    if isinstance(img, np.ndarray):
        return img if img.dtype == np.uint8 else img.astype(np.uint8)
    if isinstance(img, PILImage.Image):
        arr = np.array(img)              # RGB
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    return img

class _ResultsShim:
    class _Boxes:
        def __init__(self, xyxy_conf_cls):
            # xyxy_conf_cls: List[[x1,y1,x2,y2,conf,cls]]
            import numpy as np
            self._np = np
            if len(xyxy_conf_cls) == 0:
                self.cls = []
                self.conf = []
                self.xyxy = []
            else:
                arr = self._np.array(xyxy_conf_cls, dtype=self._np.float32)
                self.cls = arr[:, 5]
                self.conf = arr[:, 4]
                self.xyxy = arr[:, 0:4]

        def __len__(self):
            return 0 if self.xyxy == [] else self.xyxy.shape[0]

    class _Masks:
        def __init__(self):
            self.xy = []

        def __len__(self):
            return 0

    def __init__(self, xyxy_conf_cls):
        self.boxes = self._Boxes(xyxy_conf_cls)
        self.masks = self._Masks()

    def to(self, device: str):
        # Совместимость с .to('cpu')
        return self


class _TorchBackend:
    def __init__(self, seg_pt: str, tip_pt: str):
        self.seg = YOLO(seg_pt)
        self.tip = YOLO(tip_pt)

    def predict_seg(self, image: Any):
        return self.seg.predict(image)[0]

    def predict_tip(self, image: Any, conf: float = 0.5):
        return self.tip.predict(image, conf=conf)[0]


class _TrtBackend:
    def __init__(self, seg_engine: str, tip_engine: str):
        from app.models.infer_trt import TrtEngineRunner
        self.seg = TrtEngineRunner(seg_engine)
        self.tip = TrtEngineRunner(tip_engine)

    def predict_seg(self, image: Any):
        boxes = self.seg.infer_boxes(_ensure_bgr(image))
        return _ResultsShim(boxes)

    def predict_tip(self, image: Any, conf: float = 0.5):
        boxes = self.tip.infer_boxes(_ensure_bgr(image), conf_threshold=conf)
        return _ResultsShim(boxes)


class BackendFactory:
    @staticmethod
    def create() -> tuple:
        backend = os.getenv("BACKEND", "torch").strip().lower()

        seg_pt = os.getenv("SEG_PT", "models/segmentation_model.pt")
        tip_pt = os.getenv("TIP_PT", "models/tip_detector_model.pt")
        seg_engine = os.getenv("SEG_ENGINE", "models/segmentation_model_fp16.engine")
        tip_engine = os.getenv("TIP_ENGINE", "models/tip_detector_model_fp16.engine")

        if backend == "tensorrt":
            impl = _TrtBackend(seg_engine, tip_engine)
        else:
            impl = _TorchBackend(seg_pt, tip_pt)

        return impl.predict_seg, impl.predict_tip


class _SegModelAdapter:
    def __init__(self, predict_fn):
        self._predict = predict_fn

    def predict(self, image: Any):
        return [self._predict(image)]


class _TipModelAdapter:
    def __init__(self, predict_fn):
        self._predict = predict_fn

    def predict(self, image: Any, conf: Optional[float] = 0.5):
        return [self._predict(image, conf=conf if conf is not None else 0.5)]


_seg_predict, _tip_predict = BackendFactory.create()
seg_model = _SegModelAdapter(_seg_predict)
tip_model = _TipModelAdapter(_tip_predict)