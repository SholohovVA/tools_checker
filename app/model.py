from ultralytics import YOLO

def get_seg_model():
    return YOLO("models/segmentation_model.pt")

def get_tip_model():
    return YOLO("models/tip_detector_model.pt")

seg_model = get_seg_model()
tip_model = get_tip_model()