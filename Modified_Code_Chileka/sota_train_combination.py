import os.path

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset



import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
     

	loss = loss0 + loss1*0.1 + loss2*0.3
	print("l0: %3f, l1: %3f, l2: %3f\n"%(loss0.data,loss1.data,loss2.data))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss0, loss


# ------- 2. Set the arguments --------



import argparse

parser = argparse.ArgumentParser(description='Net Define')
parser.add_argument('--net', type=str, default="")
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="/home/htihe/MDPI/Data/Mosmed/Set/train/")
parser.add_argument('--tra_image_dir', type=str, default="image/")
parser.add_argument('--tra_label_dir', type=str, default="lung_mask/")

args = parser.parse_args()





# ------- 2. set the directory of training dataset --------


data_dir = args.data_dir
tra_image_dir = args.tra_image_dir
tra_label_dir = args.tra_label_dir

image_ext = '.png'
label_ext = '.png'





epoch_num = 100
batch_size_train = 4
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
tra_lbl2_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

for img_path in tra_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl2_name_list.append(data_dir + tra_label2_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train lung labels: ", len(tra_lbl_name_list))
print("train infection labels: ", len(tra_lbl2_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list_lung=tra_lbl_name_list,
    lbl_name_list_infection=tra_lbl2_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2,drop_last=True)

# ------- 3. define model --------
# define the net

def net_define(net_str):
    #12 Network
    if net_str=="couplenetc":
        from model.network_compareSOTA.couplenet import COPLENet
        return COPLENet()
    if net_str=="miniseg":
        from model.network_compareSOTA.MiniSeg import MiniSeg
        return MiniSeg()
    if net_str=="attunet":
        from model.network_compareSOTA.r2u_net import AttU_Net
        return AttU_Net()
    if net_str=="rplfunet":
        from model.network_compareSOTA.RPLFUnet import RPLFUnet
        return RPLFUnet()
    if net_str=="rplfunet":
        from model.network_compareSOTA.RPLFUnet import RPLFUnet
        return RPLFUnet()
    if net_str=="UNet_0Plus":
        from model.network_compareSOTA.UNet_0Plus import UNet
        return UNet()
    if net_str=="UNet_2D":
        from model.network_compareSOTA.unet_2D import unet_2D
        return unet_2D()
    if net_str=="UNet_2Plus":
        from model.network_compareSOTA.UNet_2Plus import UNet_2Plus
        return UNet_2Plus()
    if net_str=="UNet_3Plus":
        from model.network_compareSOTA.UNet_3Plus import UNet_3Plus
        return UNet_3Plus()
    if net_str=="UNet_Nested":
        from model.network_compareSOTA.UNet_Nested import NestedUNet
        return NestedUNet()
    if net_str=="UNet_nonlocal":
        from model.network_compareSOTA.unet_nonlocal_2D import unet_nonlocal_2D
        return unet_nonlocal_2D()
    if net_str=="SwinUNet":
        from model.network_compareSOTA.SUNet_detail import SUNet
        net = SUNet(img_size=256, patch_size=4, in_chans=3, out_chans=1,
                      embed_dim=96, depths=[8, 8, 8, 8],
                      num_heads=[8, 8, 8, 8],
                      window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                      norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                      use_checkpoint=False, final_upsample="Dual up-sample")  # .cuda()
        net.load_state_dict(torch.load(r"/home/htihe/MDPI/code/Final_Integration/ckpt/Data_ver1/SwinUNet/itr_180000_train_0.127175_tar_0.127175.pth"))
        return net
    if net_str=="cenet":
        from model.network_compareSOTA.cenet import CE_Net_
        return CE_Net_()
    if net_str=="cpfnet":
        from model.network_compareSOTA.CPFNet import CPFNet
        return CPFNet()
    if net_str=="cosupervision":
        from model.network_compareSOTA.co_supervision import RPLFUnet
        return RPLFUnet()

    if net_str=="segformer":
        from model.General_network.segformer import Segformer
        return Segformer()
    if net_str=="fcn":
        from model.General_network.fcn import FCNs,VGGNet
        vgg_model = VGGNet(requires_grad=True, remove_fc=True)
        fcn_model = FCNs(pretrained_net=vgg_model, n_class=1)
        return fcn_model
    if net_str=="pspnet":
        from model.General_network.pspnet import PSPNet
        return PSPNet()
    if net_str=="deeplabv3":
        from model.General_network.deeplabv3 import DeepLab
        return DeepLab()






net = net_define(args.net)
net.load_state_dict(torch.load(args.model_dir))
model_dir = "/home/htihe/MDPI/code/Final_Integration/ckpt/mosmed_transfer/"+args.net+'/'
if os.path.exists(model_dir)<=0:
    os.makedirs(model_dir)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer=optim.AdamW(net.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0)
# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels_lung,labels_infection = data['image'], data['label_lung'], data['label_infection']

        inputs = inputs.type(torch.FloatTensor)
        labels_lung = labels_lung.type(torch.FloatTensor)
        labels_infection= labels_infection.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_lung_v, labels_infection_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels_lung.cuda(),requires_grad=False), Variable(labels_infection.cuda(),requires_grad=False)
        else:
            inputs_v, labels_lung_v, labels_infection_v = Variable(inputs, requires_grad=False), Variable(labels_lung, requires_grad=False), Variable(labels_infection, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        infection_flow4= net(inputs_v)

        loss_infection = bce_ssim_loss(infection_flow4,labels_infection_v)

        loss=loss_infection

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % 6000 == 0:  # save model every 2000 iterations

            torch.save(net.state_dict(), model_dir + "itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
