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

        self.decode1_1 = decode_module(768, 384)

        self.decode2=decode_module(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module(384,192)

        self.decode3 = decode_module(192, 96)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module(192, 96)
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







class dualstream_swin(nn.Module):
    def __init__(self):
        super(dualstream_swin, self).__init__()

        #encoder

        self.encoder=SwinTransformer()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module(768,384)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module(768, 384)

        self.decode2=decode_module(384,192)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module(384,192)

        self.decode3 = decode_module(192, 96)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module(192, 96)
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




class dualstream_resnet50(nn.Module):
    def __init__(self):
        super(dualstream_resnet50, self).__init__()

        #encoder

        self.encoder=resnet50_encoder()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        #Decoder Chain

        self.decode1=decode_module(1024,512)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode1_1 = decode_module(1024,512)

        self.decode2=decode_module(512,256)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode2_1=decode_module(512,256)

        self.decode3 = decode_module(256,64)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decode3_1 = decode_module(128,64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)


        self.decoder4_lung=nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.sig4_lung=nn.Sigmoid()
        self.decoder4_infection = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sig4_infection = nn.Sigmoid()


        #Multiscale
        self.ms1_lung = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_lung = nn.Sigmoid()
        self.ms1_infection = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.sigms1_infection = nn.Sigmoid()
        self.ms1_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.ms2_lung = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_lung = nn.Sigmoid()
        self.ms2_infection = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.sigms2_infection = nn.Sigmoid()
        self.ms2_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.ms3_lung = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigms3_lung = nn.Sigmoid()
        self.ms3_infection = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
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
        #lung_flow3 = self.up3(lung_flow3)
        #infection_flow3 = self.up3(infection_flow3)

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




"""
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
model = dualstream().cuda()
macs, params = get_model_complexity_info(model, (3, 256,256), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import time
time1=time.time()
inputs=torch.ones((1,3,256,256)).cuda()
with torch.no_grad():
    r=model(inputs)
time2=time.time()
print("Inf time: ",time2-time1)

import time
from thop import profile
inputs=torch.ones((1,3,256,256)).cuda()
flops, params = profile(model,inputs=(inputs,))
print("Net Flops: ",flops)
print("Net Parameters Second Check: ",params)
"""