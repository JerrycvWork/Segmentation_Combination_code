import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torchvision.models
"""
net=torchvision.models.resnet50(pretrained=True)
net_list=list(net.children())[:-2]
print(len(net_list))

for i in range(len(net_list)):
    print(i)
    print(net_list[i])
"""

class resnet50_encoder(nn.Module):
    def __init__(self):
        super(resnet50_encoder, self).__init__()
        net = torchvision.models.resnet50(pretrained=True)
        net_list = list(net.children())[:-2]

        self.b1=net_list[0]
        self.b2 = net_list[1]
        self.b3 = net_list[2]
        self.b4 = net_list[3]
        self.b5 = net_list[4]
        self.b6 = net_list[5]
        self.b7 = net_list[6]
        self.b8 = net_list[7]
    def forward(self,x):
        s1=self.b4(self.b3(self.b2(self.b1(x))))
        s2=self.b5(s1)
        s3=self.b6(s2)
        s4=self.b7(s3)
        return [s1,s2,s3,s4]
