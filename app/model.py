from ultralytics import YOLO

def get_seg_model():
    return YOLO("models/seg_best_v11.pt")

def get_tip_model():
    return YOLO("models/det_best_v11.pt")

seg_model = get_seg_model()
tip_model = get_tip_model()