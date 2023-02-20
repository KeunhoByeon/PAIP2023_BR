from model import custom_resnet as resnet, custom_vgg as vgg
from model.DSF_Unet import DSF_Unet
from model.custom_baseline import Baseline


def get_model(args):
    n_filters = 60
    polar = True
    filter_width = args.filter_width

    if 'vgg' in args.model:
        model = getattr(vgg, args.model.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=args.num_classes)
    elif 'resnet' in args.model:
        model = getattr(resnet, args.model.lower())(conv_type='classical' if not polar else 'polar', kernel_size=filter_width, num_classes=args.num_classes)
    elif 'baseline' in args.model:
        polar = False
        if 'small' in args.model:
            n_filters = 20
        model = Baseline(args.input_size, n_filters, polar, args.input_channels, args.num_classes)
    elif "dsf_unet" == args.model.lower():
        model = DSF_Unet(kernel_size=filter_width, num_classes=args.num_classes)
    else:
        raise AssertionError

    return model.float()
