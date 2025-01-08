import numpy as np

def mask_to_bbox(mask: np.ndarray, image_shape: tuple) -> list:
    """
    根据掩码计算边界框。

    Args:
        mask (np.ndarray): 掩码的归一化坐标，形状为 (M, 2)。
        image_shape (tuple): 图像的形状，格式为 (H, W, C)。

    Returns:
        list: 边界框 [x_min, y_min, x_max, y_max]，以像素为单位。
    """
    H, W = image_shape[:2]

    if mask.ndim != 2 or mask.shape[1] != 2:
        raise ValueError("掩码必须是形状为 (M, 2) 的二维数组。")

    # 转换归一化坐标为像素坐标
    pixel_mask = mask.copy()
    pixel_mask[:, 0] = pixel_mask[:, 0] * W  # x 坐标
    pixel_mask[:, 1] = pixel_mask[:, 1] * H  # y 坐标

    # 计算最小和最大坐标
    x_min = np.min(pixel_mask[:, 0])
    y_min = np.min(pixel_mask[:, 1])
    x_max = np.max(pixel_mask[:, 0])
    y_max = np.max(pixel_mask[:, 1])

    # 返回边界框
    return [x_min, y_min, x_max, y_max]

if __name__=='__main__':
    mask = np.array([[0.1433, 0.145], [0.9234, 0.95551]])
    image_shape = (100, 100, 3)
    bbox = mask_to_bbox(mask, image_shape)
    print(bbox)  # [10.0, 10.0, 90.0, 90.0]