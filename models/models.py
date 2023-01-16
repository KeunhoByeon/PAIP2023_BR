import torch

from .unet_nested import NestedUNet
from models import custom_resnet as resnet, custom_vgg as vgg
from models.custom_baseline import Baseline


def get_model(args):
    n_filters = 60
    polar = True
    filter_width = args.filter_width

    if 'nestedunet' in args.model.lower():
        model = NestedUNet(num_classes=args.num_classes, input_channels=3)
    elif 'vgg' in args.model.lower():
        model = getattr(vgg, args.model.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=args.num_classes)
    elif 'resnet' in args.model.lower():
        model = getattr(resnet, args.model.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=args.num_classes)
    elif 'baseline' in args.model.lower():
        polar = False
        if 'small' in args.model:
            n_filters = 20
        model = Baseline(args.input_size, n_filters, polar, args.input_channels, args.num_classes)
    else:
        raise AssertionError

    # if torch.cuda.device_count() > 1:
    #     models = torch.nn.DataParallel(models).cuda()
    # else:
    #     models = models.to(device)

    return model.float()
