import torch

from model import custom_resnet as resnet, custom_vgg as vgg
from model.custom_baseline import Baseline


def get_model(args, data_width, n_input_channels, device):
    n_filters = 60
    polar = False
    if args.baseline_small:
        n_filters = 20
    elif args.baseline:
        pass
    else:
        polar = True
    filter_width = args.filter_width

    if 'vgg' in args.model_type:
        model = getattr(vgg, args.model_type.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=1)
    elif 'resnet' in args.model_type:
        model = getattr(resnet, args.model_type.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=1)
    else:
        model = Baseline(data_width, n_filters, polar, n_input_channels)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    return model.float()
