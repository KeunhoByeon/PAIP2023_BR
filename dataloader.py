import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.utils import resize_and_pad_image


def get_samples(data_dir, cellularity_mpp_path=None):
    if cellularity_mpp_path is not None:
        mpp_tc = {}
        with open(cellularity_mpp_path, 'r') as rf:
            header = rf.readline()
            for line in rf.readlines():
                line_split = line.replace('\n', '').split(',')
                id, mpp, tc = line_split
                mpp_tc[id] = {'mpp': mpp, 'tc': tc}

    samples = []
    for path, dir, files in os.walk(os.path.join(data_dir, 'img')):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            img_path = os.path.join(path, filename)
            ext = os.path.splitext(filename)[-1]
            id = filename.split('.')[0]
            tumor_mask_path = os.path.join(data_dir, 'mask', 'tumor', filename.replace(ext, '_tumor' + ext))
            non_tumor_mask_path = os.path.join(data_dir, 'mask', 'non_tumor', filename.replace(ext, '_nontumor' + ext))
            if not os.path.isfile(tumor_mask_path):
                print(filename)
                print('File is not exist!(tumor_mask_path): {}'.format(tumor_mask_path))
                raise FileNotFoundError
            if not os.path.isfile(non_tumor_mask_path):
                print('File is not exist!(non_tumor_mask_path): {}'.format(non_tumor_mask_path))
                raise FileNotFoundError

            data = {'id': id, 'img': img_path, 'tumor_mask': tumor_mask_path, 'non_tumor_mask': non_tumor_mask_path}
            if cellularity_mpp_path is not None:
                data['mpp'] = mpp_tc[id]['mpp']
                data['tc'] = mpp_tc[id]['tc']
            samples.append(data)

    return samples


class PAIP2023Dataset(Dataset):
    def __init__(self, data_dir: str, cellularity_mpp_path: str = None, input_size: int = None, is_train: bool = False):
        self.input_size = input_size
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Augmentation setting (Not yet implemented all)
        self.affine_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45)))
        ], random_order=True)
        self.color_seq = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            iaa.Sometimes(0.2, iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=255, squared=False)),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
            iaa.Sometimes(0.5, iaa.MultiplyHue((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.MultiplySaturation((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.2)),
            iaa.Sometimes(0.5, iaa.LogContrast((0.6, 1.4))),
        ], random_order=True)

        # if augment:
        #     self.transform = transforms.Compose([transforms.ToPILImage(),
        #                                          transforms.ColorJitter(brightness=.5, saturation=.25,
        #                                                                 hue=.1, contrast=.5),
        #                                          transforms.RandomAffine(10, (0.05, 0.05), fillcolor=255),
        #                                          transforms.RandomHorizontalFlip(.5),
        #                                          transforms.RandomVerticalFlip(.5),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
        #                                                               [0.27210504, 0.31001145, 0.2918652])])
        # else:
        #     self.transform = transforms.Compose([transforms.ToPILImage(),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
        #                                                               [0.27210504, 0.31001145, 0.2918652])])

        self.samples = get_samples(data_dir, cellularity_mpp_path)

    def __getitem__(self, index):
        data_dict = self.samples[index]

        img_path = data_dict['img']
        tumor_mask_path = data_dict['tumor_mask']
        non_tumor_mask_path = data_dict['non_tumor_mask']

        img = cv2.imread(img_path)
        tumor_mask = cv2.imread(tumor_mask_path, -1)
        non_tumor_mask = cv2.imread(non_tumor_mask_path, -1)

        mask = np.where(non_tumor_mask > 0, 2, 0)
        mask = np.where(tumor_mask > 0, 1, mask)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)
            mask = resize_and_pad_image(mask.astype(np.uint8), target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        if self.is_train:
            img = self.affine_seq.augment_image(img)
            img = self.color_seq.augment_image(img)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        mask = torch.from_numpy(mask)

        return img_path, img, mask

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_dir = './data/train'
    cellularity_mpp_path = os.path.join(data_dir, 'cellularity_mpp.csv')
    input_size = 512
    seed = 103

    import random

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    dataset = PAIP2023Dataset(data_dir=data_dir, cellularity_mpp_path=cellularity_mpp_path, input_size=input_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    for i, (img_paths, imgs, masks) in enumerate(dataloader):
        for img_path, img, mask in zip(img_paths, imgs, masks):
            original_img = cv2.imread(img_path)
            original_img = resize_and_pad_image(original_img, target_size=(input_size, input_size), keep_ratio=True, padding=True)
            img = img.numpy()
            img = img.transpose(1, 2, 0)
            img = img * std + mean
            img = img * 255.
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mask = mask.numpy()
            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor((mask * 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            cv2.imshow('T', np.hstack([original_img, img, mask]))
            cv2.waitKey(0)
