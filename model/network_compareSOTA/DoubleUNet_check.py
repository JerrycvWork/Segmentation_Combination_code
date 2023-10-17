import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
import torchvision
import numpy as np


class SuccessiveConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.successive_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.successive_conv(x)


class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SuccessiveConv(in_channels, out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x1, x2):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x, x2), dim=1)
        return self.se(self.conv(x))


class Decoder2_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SuccessiveConv(in_channels // 2 * 3, out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x1, x2, x3):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x, x2, x3), dim=1)
        return self.se(self.conv(x))


class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        if first:
            self.contracting_path = nn.Sequential(
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )
        else:
            self.contracting_path = nn.Sequential(
                nn.MaxPool2d(2),
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )

    def forward(self, x):
        return self.contracting_path(x)


class ASPP(nn.Module):
    # https://www.cnblogs.com/haiboxiaobai/p/13029920.html
    def __init__(self, in_channel=512, depth=1024):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.upsample(image_features)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class SELayer(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Double_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = torchvision.models.vgg19(pretrained=True)
        # VGG
        self.VGG_block1 = nn.Sequential(*self.VGG.features[:4])  # 64
        self.VGG_block2 = nn.Sequential(*self.VGG.features[4:9])  # 128
        self.VGG_block3 = nn.Sequential(*self.VGG.features[9:18])  # 256
        self.VGG_block4 = nn.Sequential(*self.VGG.features[18:27])  # 512
        self.VGG_block5 = nn.Sequential(*self.VGG.features[27:-1])
        # ASPP
        self.ASPP = ASPP()
        # Decoder1
        self.dec1_block1 = Decoder_Block(1024, 512)
        self.dec1_block2 = Decoder_Block(512, 256)
        self.dec1_block3 = Decoder_Block(256, 128)
        self.dec1_block4 = Decoder_Block(128, 64)
        self.dec1_conv = nn.Conv2d(64, 1, 1)
        # encoder 2
        self.enc2_block1 = Encoder_Block(3, 64, first=True)
        self.enc2_block2 = Encoder_Block(64, 128)
        self.enc2_block3 = Encoder_Block(128, 256)
        self.enc2_block4 = Encoder_Block(256, 512)
        self.enc2_block5 = Encoder_Block(512, 512)
        # decoder2
        self.dec2_block1 = Decoder2_Block(1024, 512)
        self.dec2_block2 = Decoder2_Block(512, 256)
        self.dec2_block3 = Decoder2_Block(256, 128)
        self.dec2_block4 = Decoder2_Block(128, 64)
        self.dec2_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        out1 = self.VGG_block1(x)  # 3,64
        out2 = self.VGG_block2(out1)  # 64,128
        out3 = self.VGG_block3(out2)  # 128,256
        out4 = self.VGG_block4(out3)  # 256,512
        output = self.VGG_block5(out4)
        aspp_out = self.ASPP(output)
        d1_1 = self.dec1_block1(aspp_out, out4)
        d1_2 = self.dec1_block2(d1_1, out3)
        d1_3 = self.dec1_block3(d1_2, out2)
        d1_4 = self.dec1_block4(d1_3, out1)
        d1_output = self.dec1_conv(d1_4)
        x2 = torch.matmul(x, d1_output)
        out5 = self.enc2_block1(x2)
        out6 = self.enc2_block2(out5)
        out7 = self.enc2_block3(out6)
        out8 = self.enc2_block4(out7)
        output2 = self.enc2_block5(out8)
        aspp_out2 = self.ASPP(output2)
        d2_1 = self.dec2_block1(aspp_out2, out8, out4)
        d2_2 = self.dec2_block2(d2_1, out7, out3)
        d2_3 = self.dec2_block3(d2_2, out6, out2)
        d2_4 = self.dec2_block4(d2_3, out5, out1)
        d2_output = self.dec2_conv(d2_4)
        # final_output = torch.cat((d1_output,d2_output))
        return d2_output

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

model = Double_UNet()
macs, params = get_model_complexity_info(model, (3, 256,256), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import time
time1=time.time()
inputs=torch.ones((1,3,256,256))
with torch.no_grad():
    r=model(inputs)
time2=time.time()
print("Inf time: ",time2-time1)


import time
from thop import profile
inputs=torch.ones((1,3,256,256))
flops, params = profile(model,inputs=(inputs,))
print("Net Flops: ",flops)
print("Net Parameters Second Check: ",params)

