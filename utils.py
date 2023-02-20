import cv2
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def recover_data(img, mask, mask_contour):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = img * 255.
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = mask.numpy()
    mask = mask.astype(np.uint8)

    mask_contour = mask_contour.numpy()
    mask_contour = mask_contour.astype(np.uint8)

    return img, mask, mask_contour


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)  # sigmoid를 통과한 출력이면 주석처리

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
