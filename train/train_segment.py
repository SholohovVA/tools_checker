from ultralytics import YOLO


model = YOLO('yolov11x-seg.pt')


results = model.train(
    data='/home/ak_bezborodov/PycharmProjects/hakaton/Segmentator/data_1/data.yaml',

    epochs=120,
    patience=10,
    batch=8,
    imgsz=900,
    device=[1],

    augment=True,
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.3,
    degrees=15,
    translate=0.1,
    erasing=0.2,
    scale=0.2,
    shear=2.0,
    perspective=0.0005,
    flipud=0.3,
    fliplr=0.5,
    mosaic=0.6,
    close_mosaic=30,
    copy_paste=0.3,
    copy_paste_mode='mixup',
    save=True,
    save_period=20,
    project='Segmentation_results',
    name='tools_seg_v5',
)

