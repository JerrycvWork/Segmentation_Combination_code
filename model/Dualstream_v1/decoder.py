import copy
import logging
import math

from os.path import join as pjoin
from model.Dualstream_v1.convnext import ConvNeXt_encoder
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

#import model.Dual_transformer.vit_seg_config as config

from model.Dual_transformer.Swin_transformer import SwinTransformer

import torch.nn.functional as F

from model.Dualstream_v1.fcs import load_pretrained

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = SwinTransformer()
        load_pretrained("/home/htihe/ECCV2022/FormerCode/BASNet-master/swin_tiny_patch4_window7_224.pth",self.transformer)
        #self.transformer.load_state_dict(torch.load("/home/htihe/ECCV2022/FormerCode/BASNet-master/swin_tiny_patch4_window7_224.pth")['model'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        qx = self.transformer(x)  # (B, n_patch, hidden)
        return qx

"""
torch.Size([1, 96, 56, 56])
torch.Size([1, 192, 28, 28])
torch.Size([1, 384, 14, 14])
torch.Size([1, 768, 7, 7])
"""


class convblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(convblock, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class regionattention(nn.Module):
    def __init__(self, in_channel):
        super(regionattention, self).__init__()

        # Region flow

        self.regconv = nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1)
        self.regsig=nn.Sigmoid()

        # Local Attention

        self.attconv1=nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1,dilation=1)
        self.attbn1 = nn.BatchNorm2d(in_channel)
        self.attrelu1 = nn.ReLU(inplace=True)

        self.attconv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.attbn2 = nn.BatchNorm2d(in_channel)
        self.attrelu2 = nn.ReLU(inplace=True)

        self.attconv3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=5, dilation=5)
        self.attbn3 = nn.BatchNorm2d(in_channel)
        self.attrelu3 = nn.ReLU(inplace=True)

        # Fusion Layer
        self.fusionconv=nn.Conv2d(in_channel*3, 1, kernel_size=3, stride=1, padding=1)
        self.fusionsig=nn.Sigmoid()

    def forward(self,lung_flow,infection_flow):

        lung_att=self.regsig(self.regconv(lung_flow))

        att_infection_flow=lung_att*infection_flow

        att_infection1=self.attrelu1(self.attbn1(self.attconv1(att_infection_flow)))
        att_infection2 = self.attrelu2(self.attbn2(self.attconv3(att_infection_flow)))
        att_infection3 = self.attrelu3(self.attbn2(self.attconv3(att_infection_flow)))

        att_infection_fusion=torch.cat([att_infection1,att_infection2,att_infection3],dim=1)

        att_infection=self.fusionconv(att_infection_fusion)

        fed_infection_flow=att_infection*infection_flow

        return fed_infection_flow


class decode_module(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decode_module, self).__init__()

        #Lung Region Flow
        self.regionconv1=convblock(in_channel,out_channel)

        #Lung Infection Flow
        self.infconv1 = convblock(in_channel, out_channel)

        #Region Attention
        self.att_module=regionattention(out_channel)

    def forward(self,lung_flow,infection_flow):
        lung_flow2=self.regionconv1(lung_flow)
        infection_flow2=self.infconv1(infection_flow)
        att_infection_flow=self.att_module(lung_flow2,infection_flow2)

        return lung_flow2,att_infection_flow



class dualstream(nn.Module):
    def __init__(self):
        super(dualstream, self).__init__()

        #encoder

        self.encoder=ConvNeXt_encoder()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module(384, 384)
        self.up1_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2=decode_module(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module(192,192)
        self.up2_1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3 = decode_module(192, 96)


        self.decoder4_lung=nn.Conv2d(96,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.ms2_lung = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()

    def forward(self,x):
        feat=self.encoder(x)

        # Pick final feature
        final_feat=feat[-1]
        final_feat=self.up(final_feat)


        lung_flow1,infection_flow1=self.decode1(final_feat,final_feat)
        lung_flow1=self.up1(lung_flow1)
        infection_flow1 = self.up1(infection_flow1)

        lung_flow1_1, infection_flow1_1 = self.decode1_1(lung_flow1,infection_flow1)
        lung_flow1_1 = self.up1_1(lung_flow1_1)
        infection_flow1_1 = self.up1_1(infection_flow1_1)

        lung_flow2, infection_flow2 = self.decode2(lung_flow1_1, infection_flow1_1)
        lung_flow2 = self.up2(lung_flow2)
        infection_flow2 = self.up2(infection_flow2)

        lung_flow2_1, infection_flow2_1 = self.decode2_1(lung_flow2, infection_flow2)
        lung_flow2_1 = self.up2_1(lung_flow2_1)
        infection_flow2_1 = self.up2_1(infection_flow2_1)

        lung_flow3, infection_flow3 = self.decode3(lung_flow2_1, infection_flow2_1)

        lung_flow4=self.decoder4_lung(lung_flow3)
        infection_flow4 = self.decoder4_infection(infection_flow3)
        lung_flow4=self.sig4_lung(lung_flow4)
        infection_flow4=self.sig4_infection(infection_flow4)


        #Multi_scale
        ms1_lung=self.ms1_up(self.sigms1_lung(self.ms1_lung(lung_flow1_1)))
        ms1_infection= self.ms1_up(self.sigms1_infection(self.ms1_infection(infection_flow1_1)))
        ms2_lung=self.sigms2_lung(self.ms2_lung(lung_flow2_1))
        ms2_infection= self.sigms2_infection(self.ms2_infection(infection_flow2_1))

        return lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection







class dualstream_tr(nn.Module):
    def __init__(self):
        super(dualstream_tr, self).__init__()

        #encoder

        self.encoder=SwinTransformer()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module(384, 384)
        self.up1_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2=decode_module(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module(192,192)
        self.up2_1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3 = decode_module(192, 96)


        self.decoder4_lung=nn.Conv2d(96,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.ms2_lung = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()

    def forward(self,x):
        feat=self.encoder(x)

        # Pick final feature
        final_feat=feat[-1]
        final_feat=self.up(final_feat)


        lung_flow1,infection_flow1=self.decode1(final_feat,final_feat)
        lung_flow1=self.up1(lung_flow1)
        infection_flow1 = self.up1(infection_flow1)

        lung_flow1_1, infection_flow1_1= self.decode1_1(lung_flow1,infection_flow1)
        lung_flow1_1 = self.up1_1(lung_flow1_1)
        infection_flow1_1 = self.up1_1(infection_flow1_1)

        lung_flow2, infection_flow2= self.decode2(lung_flow1_1, infection_flow1_1)
        lung_flow2 = self.up2(lung_flow2)
        infection_flow2 = self.up2(infection_flow2)

        lung_flow2_1, infection_flow2_1= self.decode2_1(lung_flow2, infection_flow2)
        lung_flow2_1 = self.up2_1(lung_flow2_1)
        infection_flow2_1 = self.up2_1(infection_flow2_1)

        lung_flow3, infection_flow3= self.decode3(lung_flow2_1, infection_flow2_1)

        lung_flow4=self.decoder4_lung(lung_flow3)
        infection_flow4 = self.decoder4_infection(infection_flow3)
        lung_flow4=self.sig4_lung(lung_flow4)
        infection_flow4=self.sig4_infection(infection_flow4)


        #Multi_scale
        ms1_lung=self.ms1_up(self.sigms1_lung(self.ms1_lung(lung_flow1_1)))
        ms1_infection= self.ms1_up(self.sigms1_infection(self.ms1_infection(infection_flow1_1)))
        ms2_lung=self.sigms2_lung(self.ms2_lung(lung_flow2_1))
        ms2_infection= self.sigms2_infection(self.ms2_infection(infection_flow2_1))


        return lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection


