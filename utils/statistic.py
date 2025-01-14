import glob
import numpy as np
from PIL import Image
import json
from collections import defaultdict

def statistic_avg_HW(img_paths):
    print(f'\033[32mnumber of leaves: {len(img_paths)}\033[0m')
    # 统计平均高度和宽度
    heights = []
    widths = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        heights.append(img.height)
        widths.append(img.width)
    print(f'average height: {np.mean(heights)}')
    print(f'average width: {np.mean(widths)}')

def get_samples_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    samples = defaultdict(int)
    for shape in data['shapes']:
        label = shape['label']
        samples[label] += 1
    return samples
    

def statistic_samples(root_dir):
    json_paths = glob.glob(f'{root_dir}/**/*.json', recursive=True)
    print(f'\033[32mnumber of images: {len(json_paths)}\033[0m')
    # 统计样本的类别分布
    total_samples = defaultdict(int)
    for json_path in json_paths:
        samples = get_samples_from_json(json_path)
        for k, v in samples.items():
            if k in ['true leaf', 'ture leaf']:
                total_samples['true leaf'] += v
            elif k in ['unwrinking', 'unwrinkling']:
                total_samples['unwrinkling'] += v
            elif k in ['wrinking', 'wrinkling']:
                total_samples['wrinkling'] += v
            else:
                total_samples[k] += v
    for k, v in total_samples.items():
        print(f'{k}: {v}')
                

if __name__=='__main__':
    img_paths = glob.glob('/home1/lujingyu/projects/AI4Science/project/datasets/contton_data_masked/images/**/*.jpg', recursive=True)
    statistic_avg_HW(img_paths)
    root_dir = '/home1/lujingyu/projects/AI4Science/project/datasets/contton_ori'
    statistic_samples(root_dir)

