'''
process the data to the format that can be used in YOLO11,
the data for identifying leaves
'''
import os
import json
import argparse
from PIL import Image

DATA_ROOT = '/home1/lujingyu/projects/AI4Science/project/datasets/contton_ori'

from utils.tools import find_files

class DataProcessor:
    def __init__(self, json_path, data_root=DATA_ROOT):
        self.json_path = json_path
        self.data = self.load_json(json_path)
        self.names = self.get_classes()
        self.data_root = data_root

    def get_classes(self):
        names = []
        assert 'shapes' in self.data.keys(), 'No shapes in the data'
        for shape in self.data['shapes']:
            names.append(shape['label'])
        unique_names = sorted(set(names))

        return unique_names
    
    @staticmethod
    def pre_class_id(label):
        # 真子叶
        if label.lower().strip() in ['true leaf', 'ture leaf']:
            return 1
        # 异常叶
        elif label.lower().strip() in ['abnormal leaf']:
            return 2
        # 冻斑
        elif label.lower().strip() in ['chill spot']:
            return 3
        return 0
    
    def load_json(self, file):
        with open(file, 'r') as f:
            return json.load(f)

    def process(self, save_path, save_img=True, save_label=True, split_type='train'):
        data = self.data
        # save image
        img_path = data['imagePath']
        if '\\' in img_path:
            img_path = img_path.replace('\\', '/')
        img_name = os.path.basename(img_path)            
        img_path = os.path.join(self.data_root, img_name)
        img = Image.open(img_path).convert('RGB')
        img_save_path = os.path.join(save_path, 'images', split_type, img_name)
        if save_img:
            img.save(img_save_path)
        first_flag = True
        for shape in data['shapes']:
            label = shape['label']
            if label.lower().strip() in ['true leaf', 'ture leaf', 'chill spot', 'abnormal leaf']:
                continue
            points = shape['points']
            # save label
            label_save_path = os.path.join(save_path, 'labels', split_type, img_name.replace('.jpg', '.txt'))
            class_id = 0
            if first_flag:
                xys = []
                for point in points:
                    x, y = point
                    x, y = x/data['imageWidth'], y/data['imageHeight']
                    xys.append(x)
                    xys.append(y)
                xys_str = ' '.join(map(str, xys))
                if save_label:
                    with open(label_save_path, 'w') as f:   
                        f.write(f'{class_id} {xys_str}\n')
                first_flag = False
            else:
                xys = []
                for point in points:
                    x, y = point
                    x, y = x/data['imageWidth'], y/data['imageHeight']
                    xys.append(x)
                    xys.append(y)
                xys_str = ' '.join(map(str, xys))
                if save_label:
                    with open(label_save_path, 'a') as f:
                        f.write(f'{class_id} {xys_str}\n')
            
        print(f'Processed {img_name}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', '-s', type=str, default='datasets/contton_data')
    parser.add_argument('--data_root', '-d', type=str, default=DATA_ROOT)
    args = parser.parse_args()

    sav_root = args.save_root
    img_root = os.path.join(sav_root, 'images')
    label_root = os.path.join(sav_root, 'labels')
    os.makedirs(sav_root, exist_ok=True)
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)
    os.makedirs(os.path.join(img_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(img_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(img_root, 'test'), exist_ok=True)
    os.makedirs(os.path.join(label_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(label_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(label_root, 'test'), exist_ok=True)


    json_files = find_files(DATA_ROOT, '.json')
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05
    train_num = int(len(json_files) * train_ratio)
    val_num = int(len(json_files) * val_ratio)
    test_num = len(json_files) - train_num - val_num
    
    for i, json_path in enumerate(json_files):
        if i < train_num:
            split_type = 'train'
        elif i < train_num + val_num:
            split_type = 'val'
        else:
            split_type = 'test'
        processor = DataProcessor(json_path)
        print(processor.get_classes())
        processor.process(sav_root, save_img=True, save_label=True, split_type=split_type)
        
    print('Done!')

