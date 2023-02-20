from model.DSF_Unet import DSF_Unet


def get_model(args):
    if "dsf_unet" == args.model.lower():
        model = DSF_Unet(kernel_size=args.filter_width, num_classes=args.num_classes)
    else:
        raise AssertionError

    return model.float()
