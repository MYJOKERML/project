import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

checkpoint = "/home1/lujingyu/projects/AI4Science/project/ckpt/sam/sam2.1_hiera_tiny.pt"
model_cfg = "/home1/lujingyu/projects/AI4Science/project/configs/model/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def read_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    return img

if __name__ == '__main__':
    img_path = "datasets/ableaves_data/images/val/IMG_1080_6.jpg"
    img = read_img(img_path)
    predictor.set_image(img)
    masks, _, _ = predictor.predict()
    print(masks)
    if masks is not None:
        print(masks.xy)