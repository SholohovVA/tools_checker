from ultralytics import YOLO

def get_model():
    return YOLO("models/best.pt")  # или ваша кастомная модель

model = get_model()