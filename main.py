# Load a model
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
# model = YOLO("/root/autodl-tmp/projects/mianhua/YOLO11/runs/segment/train9/weights/best.pt")  # load a pretrained model (recommended for training)
config_path = 'configs/data/cotton_data.yaml'
model_scale = 'yolo11s'
model = YOLO(f'{model_scale}-seg.yaml', task='segment').load(f"{model_scale}-seg.pt")  # build from YAML and transfer weights

# Train the model
imgsz = 1024
results = model.train(data=config_path, epochs=200, imgsz=imgsz, batch=8)
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
    # result.save(filename=f"result_{model_scale}_{imgsz}.jpg")  # save to disk


# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from PIL import Image
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# # 加载模型和配置
# model = YOLO("/root/projects/AI4Science/runs/segment/train5/weights/best.pt")  # load a pretrained model (recommended for training)
# # 进行预测
# image_path = '/root/projects/AI4Science/datasets/douya_data/images/val/black_ZJ045_Image_20231231214323396.jpg'
# test_results = model(image_path)

# # 使用Matplotlib叠加掩膜
# image = Image.open(image_path).convert("RGB")
# # image = image.resize((2048,2048))
# image_np = np.array(image)

# for result in test_results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     print(masks)
#     if masks is not None:
        
#         # 选择第一个掩膜的轮廓点
#         segments = masks.xy
#         # first_segment = segments[0]  # numpy.ndarray of shape [N, 2]

#         # 绘制图像和掩膜轮廓
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(image_np)
        
#         # 绘制多边形轮廓
#         for i, segment in enumerate(segments):
#             # 根据box坐标标记0/1
#             xy_center = (boxes.xywh[i][:2] + boxes.xywh[i][2:] / 2).cpu().numpy()
#             ax.text(xy_center[0], xy_center[1], str(boxes.cls[i].item()), fontsize=12, color='white')

#             if boxes.cls[i] == 0:
#                 polygon = Polygon(segment, closed=True, edgecolor='blue', fill=False, linewidth=2)
#             else:
#                 polygon = Polygon(segment, closed=True, edgecolor='red', fill=False, linewidth=2)
#             ax.add_patch(polygon)
#         # polygon = Polygon(first_segment, closed=True, edgecolor='red', fill=False, linewidth=2)
#         # ax.add_patch(polygon)

#         plt.axis('off')
#         plt.show()
#         plt.savefig('result_segment.jpg')

