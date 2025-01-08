import os
import json

def bbox_xyxy_to_xywh(bbox):
    '''
    Convert bounding box format from (xmin, ymin, xmax, ymax) to (x, y, w, h)
    '''
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def find_files(root, ext):
    '''
    find .xxx files in the root directory
    '''
    files = []
    for root, dirs, fs in os.walk(root):
        for f in fs:
            if f.endswith(ext):
                files.append(os.path.join(root, f))
    return files

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def print_json_keys(data, indent=0):
    """
    递归地打印 JSON 数据中所有字典的键。

    :param data: 解析后的 JSON 数据（可以是字典或列表）
    :param indent: 缩进级别，用于格式化输出
    """
    space = '  ' * indent  # 根据递归深度设置缩进
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{space}- {key}")
            # 递归处理值
            print_json_keys(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(f"{space}[{index}]")
            # 递归处理列表中的每个元素
            print_json_keys(item, indent + 1)
    else:
        # 如果是其他类型（如字符串、数字等），无需处理
        pass