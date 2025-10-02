import os
import argparse
from pathlib import Path

import tensorrt as trt

def parse_shape(shape_str: str):
    # "1x3x1280x960" -> (1,3,1280,960)
    return tuple(int(x) for x in shape_str.lower().split('x'))

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 workspace_mb: int = 8192,
                 min_shape: str = "1x3x1280x960",
                 opt_shape: str = "1x3x2560x1920",
                 max_shape: str = "1x3x3840x2880",
                 hw_compat: str | None = None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_data = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(trt, "HardwareCompatibilityLevel") and hasattr(config, "set_hardware_compatibility_level"):
        level_str = (hw_compat or os.getenv("HW_COMPAT", "ampere+")).lower()
        levels = {
            "none": trt.HardwareCompatibilityLevel.NONE,
            "ampere+": trt.HardwareCompatibilityLevel.AMPERE_PLUS,
            "ada+": getattr(trt.HardwareCompatibilityLevel, "ADA_PLUS", trt.HardwareCompatibilityLevel.AMPERE_PLUS),
        }
        try:
            config.set_hardware_compatibility_level(levels.get(level_str, trt.HardwareCompatibilityLevel.AMPERE_PLUS))
        except Exception:
            pass

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape_t = parse_shape(min_shape)
    opt_shape_t = parse_shape(opt_shape)
    max_shape_t = parse_shape(max_shape)
    profile.set_shape(input_tensor.name, min_shape_t, opt_shape_t, max_shape_t)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_path).write_bytes(bytes(serialized_engine))
    print(f"Saved engine: {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_onnx", default=os.getenv("SEG_ONNX", "app/models/segmentation_model.onnx"))
    parser.add_argument("--tip_onnx", default=os.getenv("TIP_ONNX", "app/models/tip_detector_model.onnx"))
    parser.add_argument("--seg_engine", default=os.getenv("SEG_ENGINE", "app/models/segmentation_model_fp16.engine"))
    parser.add_argument("--tip_engine", default=os.getenv("TIP_ENGINE", "app/models/tip_detector_model_fp16.engine"))
    parser.add_argument("--precision", default=os.getenv("PRECISION", "fp16"))
    parser.add_argument("--workspace_mb", type=int, default=int(os.getenv("WORKSPACE_MB", "8192")))
    parser.add_argument("--min", default=os.getenv("DYN_MIN", "1x3x1280x960"))
    parser.add_argument("--opt", default=os.getenv("DYN_OPT", "1x3x2560x1920"))
    parser.add_argument("--max", default=os.getenv("DYN_MAX", "1x3x3840x2880"))
    parser.add_argument("--hw_compat", default=os.getenv("HW_COMPAT", "ampere+"), help="none | ampere+ | ada+")
    args = parser.parse_args()

    fp16 = args.precision.lower() == "fp16"
    build_engine(args.seg_onnx, args.seg_engine, fp16, args.workspace_mb, args.min, args.opt, args.max, args.hw_compat)
    build_engine(args.tip_onnx, args.tip_engine, fp16, args.workspace_mb, args.min, args.opt, args.max, args.hw_compat)


if __name__ == "__main__":
    main()



