from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np
import cv2
import random
import torch

from utils.visualize import draw_masks_with_conf, draw_boxes_with_conf, draw_masks_and_boxes_with_conf

class BoxImage():
    '''
    Get the segment image from the bounding box.
    '''
    def __init__(self, img_src: np.array, xyxy: torch.tensor, conf: torch.tensor):
        '''
        args:
            img_src: np.array, shape=(H, W, 3), 图像
            xyxy: torch.tensor, shape=(1, 4), 边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
            conf: torch.tensor, shape=(1,), 置信度
        '''
        self.xyxy = xyxy.cpu().numpy().astype(np.int32)
        self.conf = conf.cpu().numpy()
        self.img_src = img_src

    @property
    def data(self):
        x1, y1, x2, y2 = self.xyxy[0]
        return self.img_src[y1:y2, x1:x2]
    



if __name__=='__main__':
    model = YOLO('ckpt/n_epoch200_best.pt')
    image_path = 'datasets/contton_data/images/test/IMG_0166.jpg'
    test_results = model(image_path)
    masks = test_results[0].masks
    boxes = test_results[0].boxes
    conf = boxes.conf
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    import time
    start_time = time.time()
    image_with_masks = draw_masks_with_conf(image_np, masks.xyn, conf, save_path='results/result_mask.jpg')
    end_time = time.time()
    print(f"绘制掩码耗时: {end_time - start_time:.4f} 秒")
    start_time = time.time()
    image_with_boxes = draw_boxes_with_conf(image_np, boxes.xyxy, conf, save_path='results/result_box.jpg')
    end_time = time.time()
    print(f"绘制边界框耗时: {end_time - start_time:.4f} 秒")
    start_time = time.time()
    image_with_conf = draw_masks_and_boxes_with_conf(image_np, masks.xyn, boxes.xyxy, conf, save_path='results/result.jpg')
    end_time = time.time()
    print(f"绘制耗时: {end_time - start_time:.4f} 秒")
    
