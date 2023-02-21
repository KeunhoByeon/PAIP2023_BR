import torch
import torch.nn as nn

from model.layers import PolarConvNd
from model.resnet_polar import *


def convrelu(in_channels, out_channels, kernel, padding):
    if kernel == 1:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            PolarConvNd(in_channels, out_channels, kernel, dimensions=2, padding=padding),
            nn.ReLU(inplace=True),
        )


class PolarDecoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x_original, x0, x1, x2, x3, x4):
        x4 = self.layer4_1x1(x4)
        x3 = self.layer3_1x1(x3)
        x2 = self.layer2_1x1(x2)
        x1 = self.layer1_1x1(x1)
        x0 = self.layer0_1x1(x0)

        x = self.upsample(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, x0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class DSF_Unet(nn.Module):
    def __init__(self, num_classes, kernel_size=3):
        super().__init__()
        self.base_model = resnet18("polar", kernel_size=kernel_size)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.decoder_seg = PolarDecoder(num_classes)
        self.decoder_instance = PolarDecoder(1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        x0 = self.layer0(input)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out_seg = self.decoder_seg(x_original, x0, x1, x2, x3, x4)
        out_instance = self.decoder_instance(x_original, x0, x1, x2, x3, x4)

        return out_seg, out_instance


if __name__ == "__main__":
    model = DSF_Unet(2, kernel_size=5)
    dummpy_input = torch.rand(1, 3, 512, 512)
    output = model(dummpy_input)
    print("output[0]", output[0].shape)
    print("output[1]", output[1].shape)
