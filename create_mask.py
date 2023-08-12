import random
import numpy as np
from skimage.measure import label, regionprops
from PIL import Image

def generate_random_mask(ratio):
    for i in range(0, 100):
        # 创建一个256x256的全零矩阵
        mask = np.zeros((256, 256), dtype=int)

        # 计算需要设置为1的元素个数
        num_ones = int(ratio * mask.size)

        # 随机生成要设置为1的像素的索引
        random_indices = np.random.choice(mask.size, num_ones, replace=False)

        # 将随机选定的索引位置设置为1
        mask.flat[random_indices] = 1

        # 将NumPy数组转换为PIL.Image
        mask = Image.fromarray(mask*255).convert("L")
    return mask
    # mask.save("data/mask/random/20/" + str(i).zfill(5) + ".png")


def generate_random_connected_block(image, min_size, max_size):
    h, w = image.shape
    while True:
        block_size = random.randint(min_size, max_size)
        x = random.randint(0, w - block_size)
        y = random.randint(0, h - block_size)
        block = np.zeros((block_size, block_size))
        image[y:y+block_size, x:x+block_size] = block
        labeled_image = label(image, connectivity=1)
        props = regionprops(labeled_image)
        if len(props) > 1:
            return image

def generate_mask(size, target_zero_ratio, min_block_size, max_block_size):
    mask = np.ones(size)
    target_zero_count = int(np.prod(size) * target_zero_ratio)
    zero_count = 0

    while zero_count < target_zero_count:
        mask = generate_random_connected_block(mask, min_block_size, max_block_size)
        zero_count = np.sum(mask == 0)

    return mask

for i in range(100):
    mask = generate_mask((256, 256), target_zero_ratio=0.2, min_block_size=1000, max_block_size=1200)
    mask.save("data/mask/1000_1200/20/" + str(i).zfill(5) + ".png")
