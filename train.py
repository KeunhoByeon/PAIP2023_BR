import argparse
import os
import random

from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import PAIP2023Dataset
from logger import Logger
from models import get_model


def val(epoch, model, criterion, val_loader, logger=None):
    model.eval()

    with torch.no_grad():
        for i, (_, inputs, targets) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == targets).item() / (inputs.shape[0] * inputs.shape[1] * inputs.shape[2]) * 100.

            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            del output, loss, acc

        if logger is not None:
            logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y%m%d%H%M%S'))


def train(epoch, model, criterion, optimizer, train_loader, logger=None):
    model.train()

    num_progress, next_print = 0, args.print_freq
    for i, (_, inputs, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        print(targets.shape)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == targets).item() / (inputs.shape[0] * inputs.shape[1] * inputs.shape[2]) * 100.

        logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})
        logger.add_history('batch', {'loss': loss.item(), 'accuracy': acc})

        num_progress += len(inputs)
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, lr=round(optimizer.param_groups[0]['lr'], 12), time=time.strftime('%Y%m%d%H%M%S'))
            next_print += args.print_freq

        del output, loss, acc

    if logger is not None:
        logger(history_key='total', epoch=epoch)


def run(args):
    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Model
    model = get_model(args)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    # CUDA
    if torch.cuda.is_available():
        model = model.cuda()

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', args.gamma, args.lr_timestep, threshold=1**(-3))

    # Dataset
    train_dataset = PAIP2023Dataset(os.path.join(args.data, 'train'), cellularity_mpp_path=os.path.join(args.data, 'train', 'cellularity_mpp.csv'), input_size=args.input_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = PAIP2023Dataset(os.path.join(args.data, 'val'), cellularity_mpp_path=None, input_size=args.input_size, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    save_dir = os.path.join(args.result, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Run training
    print('Training...')
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, logger=logger)
        if epoch % args.val_freq == 0:
            val(epoch, model, criterion, val_loader, logger=logger)
            torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(epoch)))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--models', type=str, default='nestedunet', metavar='TYPE')
    parser.add_argument('--filter-width', type=int, default=3, metavar='N', help='width of conv filters')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--pretrained', default=True, action='store_true', help='Load pretrained models.')
    # Data Arguments
    parser.add_argument('--data', default='./data', help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    parser.add_argument('--input_channels', default=3, type=int, help='image input size')
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=20, type=int, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation freq')
    parser.add_argument('--print_freq', default=1000, type=int, help='print and save frequency')
    parser.add_argument('--result', default='results', help='path to results')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--debug', default=False, action='store_true', help='debug validation')
    # ---
    parser.add_argument('--lr-timestep', type=int, default=5, metavar='M', help='Number of epochs before triggering learning rate decrease on plateau')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
