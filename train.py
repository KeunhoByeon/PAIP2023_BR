import argparse
import os
import random
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import PAIP2023Dataset
from logger import Logger
from model import get_model, DiceLoss
from utils import recover_data

warnings.filterwarnings("ignore", category=UserWarning)


def debug_image(inputs, preds_mask, preds_mask_contour, save_path):
    debug_images = []
    for img, pred_mask, pred_mask_contour in zip(inputs, preds_mask, preds_mask_contour):
        img, pred_mask, pred_mask_contour = recover_data(img, pred_mask, pred_mask_contour)

        pred_mask = cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        pred_mask_contour = cv2.cvtColor(pred_mask_contour, cv2.COLOR_GRAY2BGR)
        debug_images.append(np.hstack([img, pred_mask, pred_mask_contour]))
    cv2.imwrite(save_path, np.vstack(debug_images))


def val(args, epoch, model, criterion, val_loader, logger=None):
    model.eval()  # 모델을 평가 모드로
    criterion[0].eval()
    criterion[1].eval()

    os.makedirs(os.path.join(args.result, 'debug', str(epoch)), exist_ok=True)
    with torch.no_grad():  # Disable gradient calculation
        for i, (img_paths, inputs, masks, masks_contour) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                masks = masks.cuda()
                masks_contour = masks_contour.cuda()

            output_mask, output_mask_contour = model(inputs)
            output_mask_contour = output_mask_contour[:, 0, :, :]
            loss = criterion[0](output_mask, masks) + criterion[1](output_mask_contour, masks_contour)

            preds_mask = torch.argmax(output_mask, dim=1)
            preds_mask_contour = torch.where(output_mask_contour > 0.5, 1, 0)

            # acc_mask = torch.sum(preds_mask == masks).item() / (masks.shape[0] * masks.shape[-2] * masks.shape[-1]) * 100.
            # acc_mask_contour = torch.sum(preds_mask_contour == masks_contour).item() / (masks_contour.shape[0] * masks_contour.shape[-2] * masks_contour.shape[-1]) * 100.
            acc_mask = torch.sum(preds_mask == torch.argmax(masks, dim=1)).item() / (masks.shape[0] * masks.shape[-2] * masks.shape[-1]) * 100.
            acc_mask_contour = torch.sum(preds_mask_contour == masks_contour).item() / (masks_contour.shape[0] * masks_contour.shape[-2] * masks_contour.shape[-1]) * 100.

            logger.add_history('total', {'loss': loss.item(), 'acc_mask': acc_mask, 'acc_mask_contour': acc_mask_contour})

            save_path = os.path.join(args.result, 'debug', str(epoch), '{}.png'.format(i))
            debug_image(inputs, preds_mask, preds_mask_contour, save_path)

    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))


def train(args, epoch, model, criterion, optimizer, train_loader, logger=None):
    model.train()  # 모델을 학습 모드로
    criterion[0].train()
    criterion[1].train()

    os.makedirs(os.path.join(args.result, 'debug_train', str(epoch)), exist_ok=True)
    num_progress, next_print = 0, args.print_freq
    for i, (img_paths, inputs, masks, masks_contour) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            masks = masks.cuda()
            masks_contour = masks_contour.cuda()

        optimizer.zero_grad()
        output_mask, output_mask_contour = model(inputs)
        output_mask_contour = output_mask_contour[:, 0, :, :]
        loss = criterion[0](output_mask, masks) + criterion[1](output_mask_contour, masks_contour)
        loss.backward()
        optimizer.step()

        preds_mask = torch.argmax(output_mask, dim=1)
        preds_mask_contour = torch.where(output_mask_contour > 0.5, 1, 0)

        # acc_mask = torch.sum(preds_mask == masks).item() / (masks.shape[0] * masks.shape[-2] * masks.shape[-1]) * 100.
        # acc_mask_contour = torch.sum(preds_mask_contour == masks_contour).item() / (masks_contour.shape[0] * masks_contour.shape[-2] * masks_contour.shape[-1]) * 100.
        acc_mask = torch.sum(preds_mask == torch.argmax(masks, dim=1)).item() / (masks.shape[0] * masks.shape[-2] * masks.shape[-1]) * 100.
        acc_mask_contour = torch.sum(preds_mask_contour == masks_contour).item() / (masks_contour.shape[0] * masks_contour.shape[-2] * masks_contour.shape[-1]) * 100.

        logger.add_history('total', {'loss': loss.item(), 'acc_mask': acc_mask, 'acc_mask_contour': acc_mask_contour})
        logger.add_history('batch', {'loss': loss.item(), 'acc_mask': acc_mask, 'acc_mask_contour': acc_mask_contour})

        save_path = os.path.join(args.result, 'debug_train', str(epoch), '{}.png'.format(i))
        debug_image(inputs, preds_mask, preds_mask_contour, save_path)

        num_progress += len(inputs)
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    if logger is not None:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))


def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Model
    model = get_model(args)
    if args.resume is not None:  # resume
        model.load_state_dict(torch.load(args.resume))

    # criterion = (nn.CrossEntropyLoss(weight=torch.FloatTensor([10 / 100, 60 / 100, 80 / 100])), nn.CrossEntropyLoss(weight=torch.FloatTensor([10 / 100, 90 / 100])))
    criterion = (DiceLoss(multiclass=True), DiceLoss(multiclass=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # Dataset
    train_dataset = PAIP2023Dataset(os.path.join(args.data, 'train'), cellularity_mpp_path=os.path.join(args.data, 'train', 'cellularity_mpp.csv'), input_size=args.input_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = PAIP2023Dataset(os.path.join(args.data, 'val'), cellularity_mpp_path=None, input_size=args.input_size, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'acc_mask', 'acc_mask_contour', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train(args, epoch, model, criterion, optimizer, train_loader, logger=logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(args, epoch, model, criterion, val_loader, logger=logger)
            os.makedirs(save_dir, exist_ok=True)

            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
            torch.save(model_state_dict, os.path.join(save_dir, '{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler.step()


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='dsf_unet')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--filter-width', type=int, default=3, metavar='N', help='width of conv filters')
    parser.add_argument('--pretrained', default=True, action='store_true', help='Load pretrained model.')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    # Data Arguments
    parser.add_argument('--data', default='./data', help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--result', default='results', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
