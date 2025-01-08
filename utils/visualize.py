from PIL import Image
import numpy as np
import cv2
import torch
def draw_masks(image: np.ndarray, masks: np.ndarray, save_path: str=None):
    '''
    在图像上绘制多个掩码。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制掩码的图像
        masks: np.ndarray, 形状为 (N, M, 2)，每个掩码由 M 个归一化的 (x, y) 坐标组成，范围 [0, 1]
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_masks: np.ndarray, 形状为 (H, W, 3)，绘制了掩码的图像
    '''
    # 创建图像的副本
    image_with_masks = image.copy()

    # 获取图像的高度和宽度
    H, W = image.shape[:2]

    # 定义颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
    ]

    alpha = 0.5  # 透明度

    for i, mask in enumerate(masks):
        # 选择颜色，如果颜色列表不够则随机生成
        # color = colors[i % len(colors)] if i < len(colors) else tuple(random.randint(0, 255) for _ in range(3))
        color = colors[2]
        # 确保掩码坐标在 [0, 1] 范围内
        if np.any(mask < 0) or np.any(mask > 1):
            raise ValueError(f"Mask {i} has coordinates outside the [0, 1] range.")

        # 转换归一化坐标为像素坐标
        pixel_mask = mask.copy()
        pixel_mask[:, 0] = pixel_mask[:, 0] * W  # x 坐标
        pixel_mask[:, 1] = pixel_mask[:, 1] * H  # y 坐标

        # 转换为整数类型
        pixel_mask = pixel_mask.astype(np.int32)

        # 确保掩码是多边形闭合的
        pts = pixel_mask.reshape((-1, 1, 2))

        # 创建一个与图像相同大小的空白图层
        overlay = image_with_masks.copy()

        # 在空白图层上绘制填充的多边形
        cv2.fillPoly(overlay, [pts], color)

        # 使用addWeighted将绘制的多边形与原图进行混合
        cv2.addWeighted(overlay, alpha, image_with_masks, 1 - alpha, 0, image_with_masks)

    if save_path:
        Image.fromarray(image_with_masks).save(save_path)

    return image_with_masks

def draw_boxes(image: np.ndarray, boxes: torch.tensor, save_path: str=None):
    '''
    在图像上绘制多个边界框。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制边界框的图像
        boxes: np.ndarray, 形状为 (N, 4)，每个边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_boxes: np.ndarray, 形状为 (H, W, 3)，绘制了边界框的图像
    '''
    image_with_boxes = image.copy()
    boxes = boxes.cpu().numpy().astype(np.int32)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if save_path:
        Image.fromarray(image_with_boxes).save(save_path)

    return image_with_boxes

def draw_masks_and_boxes(image: np.ndarray, masks: np.ndarray, boxes: torch.tensor, save_path: str=None):
    '''
    在图像上绘制多个掩码和边界框。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制掩码和边界框的图像
        masks: np.ndarray, 形状为 (N, M, 2)，每个掩码由 M 个归一化的 (x, y) 坐标组成，范围 [0, 1]
        boxes: np.ndarray, 形状为 (N, 4)，每个边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_masks_and_boxes: np.ndarray, 形状为 (H, W, 3)，绘制了掩码和边界框的图像
    '''
    image_with_masks_and_boxes = image.copy()
    image_with_masks = draw_masks(image, masks)
    image_with_boxes = draw_boxes(image, boxes)
    image_with_masks_and_boxes = cv2.addWeighted(image_with_masks, 0.5, image_with_boxes, 0.5, 0)

    if save_path:
        Image.fromarray(image_with_masks_and_boxes).save(save_path)

    return image_with_masks_and_boxes

def draw_masks_and_boxes_with_conf(image: np.ndarray, masks: np.ndarray, boxes: torch.Tensor, conf: torch.Tensor, save_path: str=None):
    '''
    在图像上绘制多个掩码、边界框和置信度。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制掩码、边界框和置信度的图像
        masks: np.ndarray, 形状为 (N, M, 2)，每个掩码由 M 个归一化的 (x, y) 坐标组成，范围 [0, 1]
        boxes: torch.Tensor, 形状为 (N, 4)，每个边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
        conf: torch.Tensor, 形状为 (N,)，每个边界框的置信度
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_masks_and_boxes_and_conf: np.ndarray, 形状为 (H, W, 3)，绘制了掩码、边界框和置信度的图像
    '''
    image_with_masks_and_boxes = draw_masks_and_boxes(image, masks, boxes)
    conf = conf.cpu().numpy()
    for i, c in enumerate(conf):
        x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(np.int32)
        cv2.putText(image_with_masks_and_boxes, f"{c:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if save_path:
        Image.fromarray(image_with_masks_and_boxes).save(save_path)

    return image_with_masks_and_boxes

def draw_conf(image: np.ndarray, boxes: torch.Tensor, conf: torch.Tensor, save_path: str=None):
    '''
    在图像上绘制置信度。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制置信度的图像
        boxes: torch.Tensor, 形状为 (N, 4)，每个边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
        conf: torch.Tensor, 形状为 (N,)，每个边界框的置信度
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_conf: np.ndarray, 形状为 (H, W, 3)，绘制了置信度的图像
    '''
    image_with_conf = image.copy()
    boxes = boxes.cpu().numpy().astype(np.int32)
    conf = conf.cpu().numpy()
    for i, c in enumerate(conf):
        x, y = boxes[i, 0], boxes[i, 1]
        cv2.putText(image_with_conf, f"{c:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if save_path:
        Image.fromarray(image_with_conf).save(save_path)

    return image_with_conf

def draw_boxes_with_conf(image: np.ndarray, boxes: torch.Tensor, conf: torch.Tensor, save_path: str=None):
    '''
    在图像上绘制边界框和置信度。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制边界框和置信度的图像
        boxes: torch.Tensor, 形状为 (N, 4)，每个边界框由 (x1, y1, x2, y2) 坐标组成，坐标为整数
        conf: torch.Tensor, 形状为 (N,)，每个边界框的置信度
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_box_and_conf: np.ndarray, 形状为 (H, W, 3)，绘制了边界框和置信度的图像
    '''
    image_with_box_and_conf = image.copy()
    boxes = boxes.cpu().numpy().astype(np.int32)
    conf = conf.cpu().numpy()
    for i, c in enumerate(conf):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(image_with_box_and_conf, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image_with_box_and_conf, f"{c:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if save_path:
        Image.fromarray(image_with_box_and_conf).save(save_path)

    return image_with_box_and_conf

def draw_masks_with_conf( image: np.ndarray, masks: np.ndarray, conf: torch.Tensor, save_path: str=None):
    '''
    在图像上绘制掩码和置信度。

    参数:
        image: np.ndarray, 形状为 (H, W, 3)，要绘制掩码和置信度的图像
        masks: np.ndarray, 形状为 (N, M, 2)，每个掩码由 M 个归一化的 (x, y) 坐标组成，范围 [0, 1]
        conf: torch.Tensor, 形状为 (N,)，每个边界框的置信度
        save_path: str, 可选，结果图像的保存路径

    返回:
        image_with_mask_and_conf: np.ndarray, 形状为 (H, W, 3)，绘制了掩码和置信度的图像
    '''
    image_with_mask = draw_masks(image, masks)
    conf = conf.cpu().numpy()
    # 获取图像的高度和宽度
    H, W = image.shape[:2]
    for i, c in enumerate(conf):
        x = (masks[i][0, 0]*W).astype(np.int32)
        y = (masks[i][0, 1]*H).astype(np.int32)
        cv2.putText(image_with_mask, f"{c:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if save_path:
        Image.fromarray(image_with_mask).save(save_path)

    return image_with_mask