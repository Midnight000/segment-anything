import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
from IPython.display import display, HTML
def Valid_Ratio(image_path, mask_path):
    sam = sam_model_registry["vit_l"](checkpoint="checkpoint/sam_vit_l_0b3195.pth")
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    image = Image.open(image_path)
    image = np.array(image)
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) / 255
    searched_map = np.zeros_like(image[:, :, 0])
    segs = mask_generator.generate(image)
    sorted_segs = sorted(segs, key=(lambda x: x['area']), reverse=True)
    score = 0.0

    for seg in sorted_segs:
        seg_map = seg["segmentation"].astype(float)
        seg_map = np.clip(seg_map - searched_map, 0, 1)
        # factor代表了mask外的有效信息与其所在分割的占比
        factor = (seg_map * mask).sum() / seg_map.sum()
        if (mask * seg_map).max() + ((1 - mask) * seg_map).max() == 2:
            tmp_score = (seg_map * (1 - mask)).sum() * factor
            score = score + tmp_score
            searched_map = np.clip(seg_map + searched_map, 0, 1)
    print(score / (1-mask).sum())
    return score / (1 - mask).sum()