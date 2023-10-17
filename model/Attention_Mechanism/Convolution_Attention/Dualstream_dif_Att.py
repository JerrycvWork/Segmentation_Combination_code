import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torchvision.models

from model.Dualstream_v2.Swin_transformer import SwinTransformer
from model.Dualstream_v2.convnext import ConvNeXt_encoder
from model.Dualstream_v2.resnet50_encoder import resnet50_encoder

from model.Attention_Mechanism.Convolution_Attention.CBAM_attention import CBAM
from model.Attention_Mechanism.Convolution_Attention.SE_attention import SELayer
from model.Attention_Mechanism.Convolution_Attention.triplet_attention import TripletAttention

class convblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(convblock, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class convblock_CBAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(convblock_CBAM, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

        self.cbam=CBAM(out_channel)

    def forward(self,x):
        return self.relu(self.cbam(self.bn(self.conv(x))))


class convblock_SE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(convblock_SE, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(out_channel)

    def forward(self, x):
        return self.relu(self.se(self.bn(self.conv(x))))

class convblock_Trip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(convblock_Trip, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.trip = TripletAttention()

    def forward(self, x):
        return self.relu(self.trip(self.bn(self.conv(x))))



class decode_module_CBAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decode_module_CBAM, self).__init__()

        #Lung Region Flow
        self.regionconv1=convblock_CBAM(in_channel,out_channel)

        #Lung Infection Flow
        self.infconv1 = convblock_CBAM(in_channel, out_channel)

    def forward(self,lung_flow,infection_flow):
        lung_flow2=self.regionconv1(lung_flow)
        infection_flow2=self.infconv1(infection_flow)

        return lung_flow2,infection_flow2


class decode_module_SE(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decode_module_SE, self).__init__()

        #Lung Region Flow
        self.regionconv1=convblock_SE(in_channel,out_channel)

        #Lung Infection Flow
        self.infconv1 = convblock_SE(in_channel, out_channel)

    def forward(self,lung_flow,infection_flow):
        lung_flow2=self.regionconv1(lung_flow)
        infection_flow2=self.infconv1(infection_flow)

        return lung_flow2,infection_flow2


class decode_module_Trip(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decode_module_Trip, self).__init__()

        #Lung Region Flow
        self.regionconv1=convblock_Trip(in_channel,out_channel)

        #Lung Infection Flow
        self.infconv1 = convblock_Trip(in_channel, out_channel)

    def forward(self,lung_flow,infection_flow):
        lung_flow2=self.regionconv1(lung_flow)
        infection_flow2=self.infconv1(infection_flow)

        return lung_flow2,infection_flow2


class dualstream_CBAM(nn.Module):
    def __init__(self):
        super(dualstream_CBAM, self).__init__()

        #encoder

        self.encoder=ConvNeXt_encoder()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module_CBAM(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module_CBAM(768, 384)

        self.decode2=decode_module_CBAM(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module_CBAM(384,192)

        self.decode3 = decode_module_CBAM(192, 96)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module_CBAM(192, 96)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)


        self.decoder4_lung=nn.Conv2d(96,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.ms2_lung = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()
        self.ms2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.ms3_lung = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_lung = nn.Sigmoid()
        self.ms3_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_infection = nn.Sigmoid()

    def forward(self,x):
        feat=self.encoder(x)

        # Pick final feature
        final_feat=feat[-1]
        #final_feat=self.up(final_feat)


        lung_flow1,infection_flow1=self.decode1(final_feat,final_feat)
        lung_flow1=self.up1(lung_flow1)
        infection_flow1 = self.up1(infection_flow1)

        lung_flow1=torch.cat([lung_flow1,feat[-2]],dim=1)
        infection_flow1 = torch.cat([infection_flow1, feat[-2]], dim=1)

        lung_flow1_1, infection_flow1_1 = self.decode1_1(lung_flow1,infection_flow1)

        lung_flow2, infection_flow2 = self.decode2(lung_flow1_1, infection_flow1_1)
        lung_flow2 = self.up2(lung_flow2)
        infection_flow2 = self.up2(infection_flow2)

        lung_flow2 = torch.cat([lung_flow2, feat[-3]], dim=1)
        infection_flow2 = torch.cat([infection_flow2, feat[-3]], dim=1)

        lung_flow2_1, infection_flow2_1 = self.decode2_1(lung_flow2, infection_flow2)

        lung_flow3, infection_flow3 = self.decode3(lung_flow2_1, infection_flow2_1)
        lung_flow3 = self.up3(lung_flow3)
        infection_flow3 = self.up3(infection_flow3)

        lung_flow3 = torch.cat([lung_flow3, feat[-4]], dim=1)
        infection_flow3   = torch.cat([infection_flow3, feat[-4]], dim=1)

        lung_flow3_1, infection_flow3_1 = self.decode3_1(lung_flow3, infection_flow3)
        lung_flow3_1 = self.up4(lung_flow3_1)
        infection_flow3_1 = self.up4(infection_flow3_1)

        lung_flow4=self.decoder4_lung(lung_flow3_1)
        infection_flow4 = self.decoder4_infection(infection_flow3_1)
        lung_flow4=self.sig4_lung(lung_flow4)
        infection_flow4=self.sig4_infection(infection_flow4)


        #Multi_scale
        ms1_lung=self.ms1_up(self.sigms1_lung(self.ms1_lung(lung_flow1_1)))
        ms1_infection= self.ms1_up(self.sigms1_infection(self.ms1_infection(infection_flow1_1)))
        ms2_lung=self.ms2_up(self.sigms2_lung(self.ms2_lung(lung_flow2_1)))
        ms2_infection= self.ms2_up(self.sigms2_infection(self.ms2_infection(infection_flow2_1)))
        ms3_lung = self.sigms3_lung(self.ms3_lung(lung_flow3_1))
        ms3_infection = self.sigms3_infection(self.ms3_infection(infection_flow3_1))


        return lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection



class dualstream_SE(nn.Module):
    def __init__(self):
        super(dualstream_SE, self).__init__()

        #encoder

        self.encoder=ConvNeXt_encoder()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module_SE(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module_SE(768, 384)

        self.decode2=decode_module_SE(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module_SE(384,192)

        self.decode3 = decode_module_SE(192, 96)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module_SE(192, 96)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)


        self.decoder4_lung=nn.Conv2d(96,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.ms2_lung = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()
        self.ms2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.ms3_lung = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_lung = nn.Sigmoid()
        self.ms3_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_infection = nn.Sigmoid()

    def forward(self,x):
        feat=self.encoder(x)

        # Pick final feature
        final_feat=feat[-1]
        #final_feat=self.up(final_feat)


        lung_flow1,infection_flow1=self.decode1(final_feat,final_feat)
        lung_flow1=self.up1(lung_flow1)
        infection_flow1 = self.up1(infection_flow1)

        lung_flow1=torch.cat([lung_flow1,feat[-2]],dim=1)
        infection_flow1 = torch.cat([infection_flow1, feat[-2]], dim=1)

        lung_flow1_1, infection_flow1_1 = self.decode1_1(lung_flow1,infection_flow1)

        lung_flow2, infection_flow2 = self.decode2(lung_flow1_1, infection_flow1_1)
        lung_flow2 = self.up2(lung_flow2)
        infection_flow2 = self.up2(infection_flow2)

        lung_flow2 = torch.cat([lung_flow2, feat[-3]], dim=1)
        infection_flow2 = torch.cat([infection_flow2, feat[-3]], dim=1)

        lung_flow2_1, infection_flow2_1 = self.decode2_1(lung_flow2, infection_flow2)

        lung_flow3, infection_flow3 = self.decode3(lung_flow2_1, infection_flow2_1)
        lung_flow3 = self.up3(lung_flow3)
        infection_flow3 = self.up3(infection_flow3)

        lung_flow3 = torch.cat([lung_flow3, feat[-4]], dim=1)
        infection_flow3   = torch.cat([infection_flow3, feat[-4]], dim=1)

        lung_flow3_1, infection_flow3_1 = self.decode3_1(lung_flow3, infection_flow3)
        lung_flow3_1 = self.up4(lung_flow3_1)
        infection_flow3_1 = self.up4(infection_flow3_1)

        lung_flow4=self.decoder4_lung(lung_flow3_1)
        infection_flow4 = self.decoder4_infection(infection_flow3_1)
        lung_flow4=self.sig4_lung(lung_flow4)
        infection_flow4=self.sig4_infection(infection_flow4)


        #Multi_scale
        ms1_lung=self.ms1_up(self.sigms1_lung(self.ms1_lung(lung_flow1_1)))
        ms1_infection= self.ms1_up(self.sigms1_infection(self.ms1_infection(infection_flow1_1)))
        ms2_lung=self.ms2_up(self.sigms2_lung(self.ms2_lung(lung_flow2_1)))
        ms2_infection= self.ms2_up(self.sigms2_infection(self.ms2_infection(infection_flow2_1)))
        ms3_lung = self.sigms3_lung(self.ms3_lung(lung_flow3_1))
        ms3_infection = self.sigms3_infection(self.ms3_infection(infection_flow3_1))


        return lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection



class dualstream_Trip(nn.Module):
    def __init__(self):
        super(dualstream_Trip, self).__init__()

        #encoder

        self.encoder=ConvNeXt_encoder()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module_Trip(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module_Trip(768, 384)

        self.decode2=decode_module_Trip(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module_Trip(384,192)

        self.decode3 = decode_module_Trip(192, 96)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module_Trip(192, 96)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)


        self.decoder4_lung=nn.Conv2d(96,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.ms2_lung = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()
        self.ms2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.ms3_lung = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_lung = nn.Sigmoid()
        self.ms3_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_infection = nn.Sigmoid()

    def forward(self,x):
        feat=self.encoder(x)

        # Pick final feature
        final_feat=feat[-1]
        #final_feat=self.up(final_feat)


        lung_flow1,infection_flow1=self.decode1(final_feat,final_feat)
        lung_flow1=self.up1(lung_flow1)
        infection_flow1 = self.up1(infection_flow1)

        lung_flow1=torch.cat([lung_flow1,feat[-2]],dim=1)
        infection_flow1 = torch.cat([infection_flow1, feat[-2]], dim=1)

        lung_flow1_1, infection_flow1_1 = self.decode1_1(lung_flow1,infection_flow1)

        lung_flow2, infection_flow2 = self.decode2(lung_flow1_1, infection_flow1_1)
        lung_flow2 = self.up2(lung_flow2)
        infection_flow2 = self.up2(infection_flow2)

        lung_flow2 = torch.cat([lung_flow2, feat[-3]], dim=1)
        infection_flow2 = torch.cat([infection_flow2, feat[-3]], dim=1)

        lung_flow2_1, infection_flow2_1 = self.decode2_1(lung_flow2, infection_flow2)

        lung_flow3, infection_flow3 = self.decode3(lung_flow2_1, infection_flow2_1)
        lung_flow3 = self.up3(lung_flow3)
        infection_flow3 = self.up3(infection_flow3)

        lung_flow3 = torch.cat([lung_flow3, feat[-4]], dim=1)
        infection_flow3   = torch.cat([infection_flow3, feat[-4]], dim=1)

        lung_flow3_1, infection_flow3_1 = self.decode3_1(lung_flow3, infection_flow3)
        lung_flow3_1 = self.up4(lung_flow3_1)
        infection_flow3_1 = self.up4(infection_flow3_1)

        lung_flow4=self.decoder4_lung(lung_flow3_1)
        infection_flow4 = self.decoder4_infection(infection_flow3_1)
        lung_flow4=self.sig4_lung(lung_flow4)
        infection_flow4=self.sig4_infection(infection_flow4)


        #Multi_scale
        ms1_lung=self.ms1_up(self.sigms1_lung(self.ms1_lung(lung_flow1_1)))
        ms1_infection= self.ms1_up(self.sigms1_infection(self.ms1_infection(infection_flow1_1)))
        ms2_lung=self.ms2_up(self.sigms2_lung(self.ms2_lung(lung_flow2_1)))
        ms2_infection= self.ms2_up(self.sigms2_infection(self.ms2_infection(infection_flow2_1)))
        ms3_lung = self.sigms3_lung(self.ms3_lung(lung_flow3_1))
        ms3_infection = self.sigms3_infection(self.ms3_infection(infection_flow3_1))


        return lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection






