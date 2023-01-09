import os

import cv2
import pandas as pd
import torch


def os_walk(walk_dir, ext=None):
    if ext is None:
        ext_list = None
    elif ext == 'image' or ext == 'images' or ext == 'img' or ext == 'imgs':
        ext_list = ('.png', '.jpg', '.jpeg')
    elif isinstance(ext, list) or isinstance(ext, tuple):
        ext_list = ext
    elif isinstance(ext, str):
        ext_list = [ext]
    else:
        print("Invalid ext type: {}".format(ext))
        raise AssertionError

    for path, dir, files in os.walk(walk_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext_list is not None and ext not in ext_list and ext[1:] not in ext_list:
                continue
            yield os.path.join(path, filename)


def load_annotation(xlsx_path):
    pd_exel = pd.read_excel(xlsx_path)
    return dict(zip(pd_exel['patient'], pd_exel['EBV.positive']))


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).cpu().data.numpy()[0])
        return res


def resize_and_pad_image(img, target_size=(640, 640), keep_ratio=False, padding=False, interpolation=None):
    # 1) Calculate ratio
    old_size = img.shape[:2]
    if keep_ratio:
        ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
    else:
        new_size = target_size

    # 2) Resize image
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    # 3) Pad image
    if padding:
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img
