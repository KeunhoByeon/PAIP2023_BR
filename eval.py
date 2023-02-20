import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataloader import PAIP2023Dataset
from model import get_model
from utils import recover_data

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def run(args):
    # Model
    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # Dataset
    test_dataset = PAIP2023Dataset(os.path.join(args.data, 'val'), input_size=args.input_size, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    with torch.no_grad():
        for i, (img_paths, inputs, _, _) in tqdm(enumerate(test_loader), leave=False, desc='Evaluation', total=len(test_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            output_mask, output_mask_contour = model(inputs)
            preds_mask = torch.argmax(output_mask, dim=1)
            preds_mask_contour = torch.argmax(output_mask_contour, dim=1)

            for img, pred_mask, pred_mask_contour in zip(inputs, preds_mask, preds_mask_contour):
                img, pred_mask, pred_mask_contour = recover_data(img, pred_mask, pred_mask_contour)

                pred_mask = cv2.cvtColor((pred_mask * 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                pred_mask_contour = cv2.cvtColor(pred_mask_contour * 255, cv2.COLOR_GRAY2BGR)

                cv2.imshow('T', np.hstack([img, pred_mask, pred_mask_contour]))
                cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', type=str, default='dsf_unet')
    parser.add_argument('--filter-width', type=int, default=3, metavar='N', help='width of conv filters')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--checkpoint_name', default='20230221014044', type=str)
    parser.add_argument('--checkpoint_epoch', default=0, type=int)
    # Data Arguments
    parser.add_argument('--data', default='./data', help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--input_channels', default=3, type=int, help='image input size')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
    # Validation and Debugging Arguments
    parser.add_argument('--result', default=None, help='path to results')
    parser.add_argument('--result_tag', default='eval')
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)

    if args.checkpoint is None or len(args.checkpoint) == 0:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results/{}/checkpoints/{}.pth'.format(args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.result = './results/{}/{}/{}'.format(args.checkpoint_name, args.result_tag, args.checkpoint_epoch)
        else:
            print('Please specify result dir: {} {} {} {}'.format(args.result, args.checkpoint_name, args.result_tag, args.checkpoint_epoch))
            raise AssertionError

    args.result = os.path.expanduser(args.result)
    os.makedirs(args.result, exist_ok=True)

    run(args)
