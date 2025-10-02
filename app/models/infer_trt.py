import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


class TrtEngineRunner:

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        engine_bytes = Path(engine_path).read_bytes()
        self.engine: trt.ICudaEngine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.profile_index = 0
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) == 0:
            raise RuntimeError("Engine has no input tensors")
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.stream = cuda.Stream()

    @staticmethod
    def _preprocess(image_bgr: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image_bgr.shape[:2]
        target_w, target_h = size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        img = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW, RGB
        return img, scale, (pad_x, pad_y)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> List[int]:
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def _get_opt_input_wh(self) -> Tuple[int, int]:
        min_s, opt_s, max_s = self.engine.get_tensor_profile_shape(self.input_name, self.profile_index)

        return int(opt_s[3]), int(opt_s[2])

    def infer_boxes(self, image_bgr: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> List[List[float]]:
        orig_h, orig_w = image_bgr.shape[:2]

        target_w, target_h = self._get_opt_input_wh()
        img_chw, scale, (pad_x, pad_y) = self._preprocess(image_bgr, (target_w, target_h))

        inp = np.expand_dims(img_chw, 0).astype(self.input_dtype, copy=False)
        inp = np.ascontiguousarray(inp)

        self.context.set_input_shape(self.input_name, inp.shape)

        host_out = {}
        dev_out = {}
        out_meta = []
        for out_name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(out_name))
            if any(d == -1 for d in shape):
                raise RuntimeError(f"Dynamic output shape is unresolved for {out_name}: {shape}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(out_name))
            h = np.empty(shape, dtype=dtype, order="C")
            d = cuda.mem_alloc(h.nbytes)
            self.context.set_tensor_address(out_name, int(d))
            host_out[out_name] = h
            dev_out[out_name] = d
            out_meta.append((out_name, shape, h.dtype))

        d_in = cuda.mem_alloc(inp.nbytes)
        self.context.set_tensor_address(self.input_name, int(d_in))

        # HtoD → execute → DtoH
        cuda.memcpy_htod_async(d_in, inp, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for out_name, h in host_out.items():
            cuda.memcpy_dtoh_async(h, dev_out[out_name], self.stream)
        self.stream.synchronize()

        candidates = []
        for out_name, arr in host_out.items():
            if arr.ndim == 3:
                y = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 2:
                y = arr
            else:
                continue
            if y.shape[-1] >= 6 and y.shape[-1] <= 512:
                candidates.append((out_name, y))
        if not candidates:
            raise RuntimeError(
                f"No suitable detection-like output among: {[(n, h.shape) for n, h in host_out.items()]}")

        chosen_name, y = max(candidates, key=lambda kv: kv[1].shape[0])

        if y.size == 0:
            return []

        boxes_xywh = y[:, 0:4].astype(np.float32)
        obj = y[:, 4].astype(np.float32)
        cls_logits = y[:, 5:].astype(np.float32)
        cls_ids = cls_logits.argmax(axis=1)
        cls_conf = cls_logits.max(axis=1)
        scores = obj * cls_conf

        mask = scores >= conf_threshold
        if not np.any(mask):
            return []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        # back to xyxy оригинального изображения
        cx, cy, w, h = boxes_xywh.T
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes_xyxy, scores, iou_threshold)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        results: List[List[float]] = []
        for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
            # clip
            x1 = float(max(0.0, min(x1, orig_w - 1)))
            y1 = float(max(0.0, min(y1, orig_h - 1)))
            x2 = float(max(0.0, min(x2, orig_w - 1)))
            y2 = float(max(0.0, min(y2, orig_h - 1)))
            results.append([x1, y1, x2, y2, float(sc), int(cid)])
        return results