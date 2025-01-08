from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import json
import os
import typing as tp
import torch
import cv2

from utils.visualize import draw_masks_with_conf
from collections import defaultdict

def extract_masked_image(image: np.ndarray, masks: tp.List[np.ndarray]) -> np.ndarray:
    """
    创建一幅仅保留掩码部分的空白图像。

    Args:
        image (np.ndarray): 原始图像，形状为 (H, W, 3)。
        masks (List[np.ndarray]): 掩码列表，每个掩码是归一化的 (x, y) 坐标数组，范围 [0, 1]。

    Returns:
        np.ndarray: 仅保留掩码部分的图像，其余部分为黑色。
    """
    H, W = image.shape[:2]
    # 创建一个与原始图像相同大小的黑色图像
    masked_image = np.zeros_like(image)

    for i, mask in enumerate(masks):
        # 检查掩码形状
        if mask.ndim != 2 or mask.shape[1] != 2:
            raise ValueError("每个掩码必须是形状为 (M, 2) 的二维数组。")

        # 转换归一化坐标为像素坐标
        pixel_mask = mask.copy()
        pixel_mask[:, 0] = pixel_mask[:, 0] * W  # x 坐标
        pixel_mask[:, 1] = pixel_mask[:, 1] * H  # y 坐标

        # 将坐标转换为整数类型
        pixel_mask = pixel_mask.astype(np.int32)

        # 确保掩码形成闭合多边形
        pts = pixel_mask.reshape((-1, 1, 2))

        # 创建一个空白掩码图层
        single_mask = np.zeros((H, W), dtype=np.uint8)

        # 绘制填充的多边形
        cv2.fillPoly(single_mask, [pts], 255)

        # 创建3通道掩码
        single_mask_3ch = cv2.merge([single_mask, single_mask, single_mask])

        # 将原始图像的掩码部分复制到空白图像
        masked_image = cv2.bitwise_or(masked_image, cv2.bitwise_and(image, single_mask_3ch))

    return masked_image

if __name__=='__main__':
    ckpt_path = 'ckpt/m_epoch200_best.pt'
    model = YOLO(ckpt_path)
    img_paths = glob.glob('datasets/contton_data/images/test/*.jpg')
    test_path = img_paths[0]
    print(f'testing image: {test_path}...')
    results = model(img_paths)
    save_root = 'data/pure_leaves'

    for i, result in enumerate(results):
        img_path = img_paths[i]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_name = os.path.basename(img_path)
        model_name = os.path.basename(ckpt_path).split('.')[0]
        os.makedirs(f'results/{model_name}', exist_ok=True)
        result.save(filename=f'results/{model_name}/result_{img_name}') # 保存结果图像
        # 从结果中提取掩码、边界框和置信度
        masks = result.masks
        boxes = result.boxes
        conf = result.boxes.conf
        # 保留置信度大于 0.79 的结果
        threshold = 0.79
        valid_idx = conf > threshold
        masks = masks[valid_idx]
        boxes = boxes[valid_idx]
        conf = conf[valid_idx]

        # 储存元信息
        meta_info = {
            'image_path': img_path,
            'height': img.shape[0],
            'width': img.shape[1],
            'leaves': [None] * len(masks),
            # 'masks': [mask.tolist() for mask in masks.xyn],            
            # 'boxes': boxes.xyxy.cpu().numpy().tolist(), 
            # 'conf': conf.cpu().numpy().tolist(),
        }
        for i, mask in enumerate(masks):
            meta_info['leaves'][i] = {
                'index': i,
                'confidence': conf[i].cpu().numpy().tolist(),
                'mask': mask.xyn[0].tolist(),
                'box': boxes[i].xyxy.cpu().numpy().tolist(),
            }
        # 将元信息写入 JSON 文件
        meta_info_dir = os.path.join(save_root, 'meta_info')
        os.makedirs(meta_info_dir, exist_ok=True)

        with open(os.path.join(meta_info_dir, img_name.replace('.jpg', '.json')), 'w') as f:
            json.dump(meta_info, f, indent=4)  # 使用缩进便于阅读
        # draw_masks_with_conf(img, masks.xyn, conf, save_path='drawn.jpg')
        masked_img = extract_masked_image(img, masks.xyn)
        os.makedirs(os.path.join(save_root, 'masked_images'), exist_ok=True)
        Image.fromarray(masked_img).save(os.path.join(save_root, 'masked_images', f'{img_name.split(".")[0]}_masked.jpg'))
