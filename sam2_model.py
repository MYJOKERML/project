import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import hydra

config_dir="configs/model/sam2.1/"
model_cfg = "sam2.1_hiera_t.yaml"
sam2_checkpoint = "ckpt/sam/sam2.1_hiera_tiny.pt"

# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module(config_dir, version_base='1.2')

# this should work now
sam2_model = build_sam2(model_cfg, sam2_checkpoint)


predictor = SAM2ImagePredictor(sam2_model)

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