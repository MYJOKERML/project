from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
# model = YOLO("/home1/lujingyu/projects/AI4Science/project/runs/segment/train3/weights/best.pt")  # load a pretrained model (recommended for training)
config_path = '/home1/lujingyu/projects/AI4Science/project/configs/data/cotton_data.yaml'
model_scale = 'yolo11m'
model = YOLO(f'{model_scale}-seg.yaml', task='segment').load(f"{model_scale}-seg.pt")  # build from YAML and transfer weights

# Train the model
imgsz = 1024
results = model.train(data=config_path, epochs=200, imgsz=imgsz, batch=6)
test_results = model('datasets/contton_data/images/test/IMG_0166.jpg')
# Process results list
for result in test_results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(boxes.conf)
    # result.show()  # display to screen
    result.save(filename=f"../results/result_{model_scale}_{imgsz}.jpg")  # save to disk