import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import Ratio
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
from IPython.display import display, HTML
for masks_path in os.listdir("data/mask_one/"):
    gts_path = "data/gt/"
    masks_path = os.path.join("data/mask_one/", masks_path)
    results_path = "results/mask_one/" + str.replace(masks_path, "data/mask_one/", "")
    gts = os.listdir(gts_path)
    masks = os.listdir(masks_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    file = open(os.path.join(results_path, "result.txt"), "w")
    total_score = 0
    for gt_mask in zip(gts, masks):
        gt_path = os.path.join(gts_path, gt_mask[0])
        mask_path = os.path.join(masks_path, gt_mask[1])
        score = Ratio.Valid_Ratio(gt_path, mask_path)
        total_score += score
        file.write(gt_mask[0] + ":" + str(round(score, 5)) + "\n")
    file.write("average: " + str(round(total_score / len(gts), 5)))