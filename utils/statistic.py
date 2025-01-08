import glob
import numpy as np
from PIL import Image

if __name__=='__main__':
    img_paths = glob.glob('/home1/lujingyu/projects/AI4Science/project/datasets/contton_data_masked/images/**/*.jpg', recursive=True)
    print(f'number of images: {len(img_paths)}')
    # 统计平均高度和宽度
    heights = []
    widths = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        heights.append(img.height)
        widths.append(img.width)
    print(f'average height: {np.mean(heights)}')
    print(f'average width: {np.mean(widths)}')
