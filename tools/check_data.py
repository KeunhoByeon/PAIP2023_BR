import os

import cv2
import numpy as np

from data.utils import os_walk


def get_data_dict(data_dir, cellularity_mpp_path):
    mpps = []
    mpp_tc = {}
    with open(cellularity_mpp_path, 'r') as rf:
        header = rf.readline()

        for line in rf.readlines():
            line_split = line.replace('\n', '').split(',')
            id, mpp, tc = line_split
            mpp_tc[id] = (mpp, tc)
            mpps.append(mpp)

    data = {}
    for img_path in os_walk(os.path.join(data_dir, 'img'), 'images'):
        filename = os.path.basename(img_path)
        ext = os.path.splitext(filename)[-1]
        id = filename.split('.')[0]
        tumor_mask_path = os.path.join(data_dir, 'mask', 'tumor', filename.replace(ext, '_tumor' + ext))
        non_tumor_mask_path = os.path.join(data_dir, 'mask', 'non_tumor', filename.replace(ext, '_nontumor' + ext))
        if not os.path.isfile(tumor_mask_path):
            print('File is not exist!: {}'.format(tumor_mask_path))
            raise FileNotFoundError
        if not os.path.isfile(non_tumor_mask_path):
            print('File is not exist!: {}'.format(non_tumor_mask_path))
            raise FileNotFoundError

        data[id] = {'img': img_path, 'tumor_mask': tumor_mask_path, 'non_tumor_mask': non_tumor_mask_path, 'mpp': mpp_tc[id][0], 'tc': mpp_tc[id][1]}

    return data


if __name__ == '__main__':
    data_dir = '../data/train'
    cellularity_mpp_path = os.path.join(data_dir, 'cellularity_mpp.csv')

    see_label = False
    see_seperate = False

    data = get_data_dict(data_dir, cellularity_mpp_path)
    for id, data_dict in data.items():
        img_path = data_dict['img']
        tumor_mask_path = data_dict['tumor_mask']
        non_tumor_mask_path = data_dict['non_tumor_mask']
        mpp = data_dict['mpp']
        tc = data_dict['tc']

        img = cv2.imread(img_path)
        tumor_mask = cv2.imread(tumor_mask_path, -1)
        non_tumor_mask = cv2.imread(non_tumor_mask_path, -1)

        num_tumor = np.max(tumor_mask)
        num_non_tumor = np.max(non_tumor_mask)

        if see_label:
            tumor_mask = np.where(tumor_mask > 0, (tumor_mask + 64) * 255 / np.max(tumor_mask + 64), 0)
            non_tumor_mask = np.where(non_tumor_mask > 0, (non_tumor_mask + 64) * 255 / np.max(non_tumor_mask + 64), 0)
        else:
            tumor_mask = np.where(tumor_mask > 0, 255, 0)
            non_tumor_mask = np.where(non_tumor_mask > 0, 255, 0)

        conflict = np.where(np.where(tumor_mask > 0, 1, 0) + np.where(non_tumor_mask > 0, 1, 0) > 1, 255, 0)
        tumor_mask = np.where(conflict, 0, tumor_mask)
        non_tumor_mask = np.where(conflict, 0, non_tumor_mask)

        tumor_mask = np.stack([np.zeros_like(tumor_mask), tumor_mask, np.zeros_like(tumor_mask)], axis=-1)
        non_tumor_mask = np.stack([non_tumor_mask, np.zeros_like(non_tumor_mask), np.zeros_like(non_tumor_mask)], axis=-1)
        conflict = np.stack([np.zeros_like(conflict), np.zeros_like(conflict), conflict], axis=-1)
        overlap = tumor_mask + non_tumor_mask + conflict
        img_overlap = np.clip(img + overlap, 0, 255).astype(np.uint8)

        cv2.putText(img, "MPP: {}".format(mpp), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, "TC: {} ({})".format(tc, round(num_tumor / (num_tumor + num_non_tumor) * 100, 2)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        if see_seperate:
            output = np.hstack([img, img_overlap, overlap, tumor_mask, non_tumor_mask]).astype(np.uint8)
        else:
            output = np.hstack([img, img_overlap, overlap]).astype(np.uint8)

        cv2.imshow('T', output)
        key = cv2.waitKey(0)

        if key == ord('q'):
            exit(0)
