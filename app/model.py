from ultralytics import YOLO

def get_model():
#    return YOLO("models/best.pt")
    return YOLO("models/best_konec.pt")

model = get_model()