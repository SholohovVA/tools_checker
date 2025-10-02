import os
import argparse
from pathlib import Path

import onnx
import onnx.checker as onnx_checker
import onnxruntime as ort
from ultralytics import YOLO

import sys

def _yolo_export_safely(model, **kwargs):
    argv_bak = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return model.export(**kwargs)
    finally:
        sys.argv = argv_bak


def export_one(pt_path: str, onnx_path: str, opset: int = 18, dynamic: bool = True, imgsz=(2560, 1920)):
    model = YOLO(pt_path)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    _yolo_export_safely(
    model,
    format="onnx",
    opset=22,          
    dynamic=True,
    imgsz=[2560, 1920]
    )
    auto_path = Path(pt_path).with_suffix(".onnx")
    if auto_path.exists() and auto_path != onnx_file:
        auto_path.replace(onnx_file)

    onnx_model = onnx.load(str(onnx_file))
    onnx_checker.check_model(onnx_model)
    return str(onnx_file)


def quick_validate(onnx_path: str, resolutions=((1280, 960), (2560, 1920), (3840, 2880))):
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    import numpy as np
    for w, h in resolutions:
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        outs = session.run(None, {input_name: dummy})
        assert isinstance(outs, list) and len(outs) >= 1, "ONNX runtime returned no outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_pt", default=os.getenv("SEG_PT", "app/models/segmentation_model.pt"))
    parser.add_argument("--tip_pt", default=os.getenv("TIP_PT", "app/models/tip_detector_model.pt"))
    parser.add_argument("--seg_onnx", default=os.getenv("SEG_ONNX", "app/models/segmentation_model.onnx"))
    parser.add_argument("--tip_onnx", default=os.getenv("TIP_ONNX", "app/models/tip_detector_model.onnx"))
    parser.add_argument("--opset", type=int, default=int(os.getenv("OPSET", "17")))
    args = parser.parse_args()

    seg_onnx = export_one(args.seg_pt, args.seg_onnx, opset=args.opset)
    tip_onnx = export_one(args.tip_pt, args.tip_onnx, opset=args.opset)

    quick_validate(seg_onnx)
    quick_validate(tip_onnx)

    print(f"Exported and validated: {seg_onnx}, {tip_onnx}")


if __name__ == "__main__":
    main()



