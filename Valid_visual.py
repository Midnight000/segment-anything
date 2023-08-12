import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def Valid_map(image_path, mask_path):
    sam = sam_model_registry["vit_l"](checkpoint="checkpoint/sam_vit_l_0b3195.pth")
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    image = Image.open(image_path)
    image = np.array(image)
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) / 255
    searched_map = np.zeros_like(image[:, :, 0]).astype(float)
    valid_map = np.zeros_like(image[:, :, 0]).astype(float)
    segs = mask_generator.generate(image)
    sorted_segs = sorted(segs, key=(lambda x: x['area']), reverse=True)
    score = 0.0
    for seg in sorted_segs:
        seg_map = seg["segmentation"].astype(float)
        seg_map = np.clip(seg_map - searched_map, 0, 1)
        # factor代表了mask外的有效信息与其所在分割的占比
        if (mask * seg_map).max() + ((1 - mask) * seg_map).max() == 2:
            factor = (seg_map * mask).sum() / seg_map.sum()
            valid_map += (1 - mask) * seg_map * factor
            tmp_score = (seg_map * (1 - mask)).sum() * factor
            score = score + tmp_score
            searched_map = np.clip(seg_map + searched_map, 0, 1)
    return valid_map


valid_map = Valid_map("data/gt/00009.png", "mask.png")
plt.figure(figsize=(3, 3))
plt.imshow(valid_map, cmap="plasma")
plt.show()