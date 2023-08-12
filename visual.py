import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

sam = sam_model_registry["vit_l"](checkpoint="checkpoint/sam_vit_l_0b3195.pth")
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)
image = Image.open("data/gt/00074.png")
image = np.array(image)
masks = mask_generator.generate(image)
plt.figure(figsize=(3, 3))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()