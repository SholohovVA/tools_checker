from ultralytics import YOLO

model = YOLO('yolo11l.pt')

augment_hyp = {
    'hsv_h': 0.015,
    'hsv_s': 0.3,
    'hsv_v': 0.3,
    'degrees': 90.0,
    'translate': 0.05,
    'scale': 0.2,
    'shear': 3.0,
    'perspective': 0.0005,
    'flipud': 0.25,
    'fliplr': 0.25,
    'mosaic': 0.5,
    'mixup': 0.05,
}

# Обучение
results = model.train(
    data='/home/ak_bezborodov/PycharmProjects/hakaton/Detector/dataset_3/data.yaml',
    epochs=150,
    patience=15,
    imgsz=1200,
    batch=8,
    save=True,
    save_period=20,
    device=[1],
    project='Detection_Results',
    name='detection_konchiki_v3_cls_l',
    hsv_h=augment_hyp['hsv_h'],
    hsv_s=augment_hyp['hsv_s'],
    hsv_v=augment_hyp['hsv_v'],
    degrees=augment_hyp['degrees'],
    translate=augment_hyp['translate'],
    scale=augment_hyp['scale'],
    shear=augment_hyp['shear'],
    perspective=augment_hyp['perspective'],
    flipud=augment_hyp['flipud'],
    fliplr=augment_hyp['fliplr'],
    mosaic=augment_hyp['mosaic'],
    mixup=augment_hyp['mixup'],
    rect=False,
    cls=2.5,
)