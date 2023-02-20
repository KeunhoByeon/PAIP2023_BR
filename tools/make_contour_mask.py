import os
import cv2
import numpy as np
from tqdm import tqdm

base_dir = '../data/train/mask'
save_dir = '../data/train/mask_contour'

if __name__ == '__main__':
    mask_paths = []
    for path, dir, files in os.walk(os.path.join(base_dir)):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            mask_path = os.path.join(path, filename)
            mask_paths.append(mask_path)

    for mask_path in tqdm(mask_paths):
        mask = cv2.imread(mask_path, -1)
        mask = cv2.resize(mask, (512, 512))
        contour_mask = np.zeros_like(mask)

        for i in np.unique(mask):
            if i == 0:
                continue
            contours, hierachy = cv2.findContours((mask == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_mask = cv2.drawContours(contour_mask, contours, -1, 255, 1)
        contour_mask = contour_mask.astype(np.uint8)

        save_path = mask_path.replace(base_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, contour_mask)
