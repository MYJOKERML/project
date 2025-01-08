from modules.mask_leaves import extract_masked_image
from utils.mask_to_bbox import mask_to_bbox
import json
import os
import glob
from PIL import Image
import numpy as np

DATAROOT = '/home1/lujingyu/projects/AI4Science/project/datasets/contton_ori'
class DataProcessor:
    def __init__(self, json_path, data_root):
        self.json_path = json_path
        self.data = self.load_json(json_path)
        self.data_root = data_root
    
    def load_json(self, file):
        with open(file, 'r') as f:
            return json.load(f)
    
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
    
    @staticmethod
    def classes(self, class_id):
        if class_id == 1:
            return 'true leaf'
        elif class_id == 2:
            return 'abnormal leaf'
        elif class_id == 3:
            return 'chill spot'
        return 'unknown'


    def save_masked_image(self, save_root):
        data = self.data
        img_path = data['imagePath']
        if '\\' in img_path:
            img_path = img_path.replace('\\', '/')
        img_name = os.path.basename(img_path)            
        img_path = os.path.join(self.data_root, img_name)
        height, width = data['imageHeight'], data['imageWidth']

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        masks = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if label.lower().strip() in ['true leaf', 'ture leaf', 'chill spot', 'abnormal leaf']:
                continue
            xys = np.array(points)
            xys[:, 0] = xys[:, 0] / width
            xys[:, 1] = xys[:, 1] / height
            masks.append(xys)
        
        masked_img = extract_masked_image(img, masks)
        os.makedirs(os.path.join(save_root, 'masked_images'), exist_ok=True)
        Image.fromarray(masked_img).save(os.path.join(save_root, 'masked_images', f'{img_name}'))

    def save_masked_image_with_meta_info(self, save_root):
        data = self.data
        img_path = data['imagePath']
        if '\\' in img_path:
            img_path = img_path.replace('\\', '/')
        img_name = os.path.basename(img_path)            
        img_path = os.path.join(self.data_root, img_name)
        height, width = data['imageHeight'], data['imageWidth']

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        masks = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if label.lower().strip() in ['true leaf', 'ture leaf', 'chill spot', 'abnormal leaf']:
                continue
            xys = np.array(points)
            xys[:, 0] = xys[:, 0] / width
            xys[:, 1] = xys[:, 1] / height
            masks.append(xys)
        
        masked_img = extract_masked_image(img, masks)
        os.makedirs(os.path.join(save_root, 'masked_images'), exist_ok=True)
        Image.fromarray(masked_img).save(os.path.join(save_root, 'masked_images', f'{img_name}'))

        meta_info = {
            'ori_image_path': img_path,
            'mask_image_path': os.path.join(save_root, 'masked_images', f'{img_name}'),
            'height': height,
            'width': width,
            'leaves': [None] * len(masks),
        }
        for i, mask in enumerate(masks):
            meta_info['leaves'][i] = {
                'index': i,
                'mask': mask.tolist(),
                'box': mask_to_bbox(mask, (height, width)),
            }
        
        meta_info_dir = os.path.join(save_root, 'meta_info')
        os.makedirs(meta_info_dir, exist_ok=True)
        meta_info_path = os.path.join(meta_info_dir, img_name.replace('.jpg', '.json'))
        with open(meta_info_path, 'w') as f:
            json.dump(meta_info, f, indent=4)
        
        return meta_info_path
    
    def make_cropped_data(self, meta_info_json_path, save_root, save_img=True, save_label=True, split_type='train'):
        meta_info = self.load_json(meta_info_json_path)
        H, W = meta_info['height'], meta_info['width']
        img_path = meta_info['mask_image_path']
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        for leaf in meta_info['leaves']:
            bbox_xyxy = np.array(leaf['box'], dtype=np.int32)
            assert bbox_xyxy[-2] < W and bbox_xyxy[-1] < H, f'BOUNDING ERROR! bbox: {bbox_xyxy}, W: {W}, H: {H}'
            cropped_img = img[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
            os.makedirs(os.path.join(save_root, 'images', split_type), exist_ok=True)
            Image.fromarray(cropped_img).save(os.path.join(save_root, 'images', split_type, f'{img_name.split(".")[0]}_{leaf["index"]}.jpg'))
            os.makedirs(os.path.join(save_root, 'labels', split_type), exist_ok=True)
            class_ids, masks = self.get_labels(leaf)
            first_flag = True
            for class_id, mask in zip(class_ids, masks):
                label_save_path = os.path.join(save_root, 'labels', split_type, f'{img_name.split(".")[0]}_{leaf["index"]}.txt')
                if first_flag:
                    if save_label:
                        with open(label_save_path, 'w') as f:   
                            f.write(f'{class_id} {mask}\n')
                    first_flag = False
                else:
                    if save_label:
                        with open(label_save_path, 'a') as f:
                            f.write(f'{class_id} {mask}\n')

        print(f'{img_name} has been processed. Total {len(meta_info["leaves"])} leaves.')
    def save_config(self, save_root):
        config = {
            'classes': ['true leaf', 'abnormal leaf', 'chill spot'],
            'num_classes': 3,
            'train': os.path.join(save_root, 'images', 'train'),
            'val': os.path.join(save_root, 'images', 'val'),
            'train_label': os.path.join(save_root, 'labels', 'train'),
            'val_label': os.path.join(save_root, 'labels', 'val')
        }
        with open(os.path.join(save_root, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def normalize(self, x, y, W, H):
        return x/W, y/H

    def get_labels(self, leaf):
        bbox = leaf['box']
        x1, y1, x2, y2 = bbox
        class_ids = []
        masks = []
        for shape in self.data['shapes']:
            label = shape['label']
            class_id = self.pre_class_id(label)
            if class_id == 0:
                continue
            points = shape['points']
            xys_ori = np.array(points)
            xys = np.zeros_like(xys_ori)
            xys[:, 0] = xys_ori[:, 0] / self.data['imageWidth']
            xys[:, 1] = xys_ori[:, 1] / self.data['imageHeight']
            sub_bbox = mask_to_bbox(xys, (self.data['imageHeight'], self.data['imageWidth']))

            if sub_bbox[0] >= x1 and sub_bbox[1] >= y1 and sub_bbox[2] <= x2 and sub_bbox[3] <= y2:
                class_ids.append(class_id-1)
                # mask 是相对于 bbox 的坐标
                xys_rel = np.zeros_like(xys_ori)
                xys_rel[:, 0] = (xys_ori[:, 0] - x1) / (x2 - x1)
                xys_rel[:, 1] = (xys_ori[:, 1] - y1) / (y2 - y1)
                xys_rel_str = ' '.join(map(str, xys_rel.flatten()))
                masks.append(xys_rel_str)
            
        return class_ids, masks
        
if __name__=='__main__':
    save_root = 'datasets/contton_data_masked'
    os.makedirs(save_root, exist_ok=True)
    json_files = glob.glob(os.path.join(DATAROOT, '*.json'))
    # test_json = '/home1/lujingyu/projects/AI4Science/project/datasets/contton_ori/IMG_0167.json'
    # data_processor = DataProcessor(test_json, DATAROOT)
    # meta_info_path = data_processor.save_masked_image_with_meta_info(save_root)
    # data_processor.make_cropped_data(meta_info_path, save_root, split_type='train')
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05
    train_num = int(len(json_files) * train_ratio)
    val_num = int(len(json_files) * val_ratio)
    test_num = len(json_files) - train_num - val_num
    for i, json_file in enumerate(json_files):
        if i < train_num:
            split_type = 'train'
        elif i < train_num + val_num:
            split_type = 'val'
        else:
            split_type = 'test'
        data_processor = DataProcessor(json_file, DATAROOT)
        meta_info_path = data_processor.save_masked_image_with_meta_info(save_root)
        data_processor.make_cropped_data(meta_info_path, save_root, split_type=split_type)