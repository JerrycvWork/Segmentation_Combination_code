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

from chaindata_loader import Rescale
from chaindata_loader import RescaleT
from chaindata_loader import RandomCrop
from chaindata_loader import CenterCrop
from chaindata_loader import ToTensor
from chaindata_loader import ToTensorLab
from chaindata_loader import SalObjDataset

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
prob_loss=nn.BCELoss(size_average=True)

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

	loss = loss0 + loss1*0.1 + loss2*0.1
	print("l0: %3f, l1: %3f, l2: %3f\n"%(loss0.data,loss1.data,loss2.data))

	return loss0, loss



# ------- 2. set the directory of training dataset --------

import argparse

parser = argparse.ArgumentParser(description='Net Define')
parser.add_argument('--net', type=str, default="")
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="Sample_Dataset/Train/")
parser.add_argument('--tra_image_dir', type=str, default="image/")
parser.add_argument('--tra_label_dir', type=str, default="lung_mask/")
parser.add_argument('--tra_label2_dir', type=str, default="infection_mask/")

## Training Parameters 
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--multi_scale_factor', type=float, default=0.1)


args = parser.parse_args()





data_dir = args.data_dir
tra_image_dir = args.tra_image_dir
tra_label_dir = args.tra_label_dir
tra_label2_dir = args.tra_label2_dir

image_ext = '.png'
label_ext = '.png'




epoch_num = args.epoch
batch_size_train = args.batch_size

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
        RescaleT(args.image_size),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)




# ------- 3. define model --------
# define the net




def net_define(net_str):
    #Our Network
    if net_str=="dualstream_v1":
        from model.Dualstream_v1.decoder import dualstream as dualstream_v1
        return dualstream_v1()
    if net_str=="dualstream_v2":
        from model.Dualstream_v2.conv_decoder import dualstream as dualstream_v2
        return dualstream_v2()
    
    ## Different Attention
    if net_str=="MS_guide_attention":
        from model.Attention_Mechanism.Network_Attention.MS_guide_attention import DAF_stack
        return DAF_stack()
    if net_str=="pranet":
        from model.Attention_Mechanism.Network_Attention.Pra_attention import PraNet
        return PraNet()
    if net_str == "pyramid":
        from model.Attention_Mechanism.Network_Attention.Pytamid_Attention import ResNet50, Classifier, PAN, Mask_Classifier,Seg_total
        return Seg_total()
    if net_str=="dualstream_CBAM":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_CBAM()
    if net_str=="dualstream_SE":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_SE()
    if net_str=="dualstream_Trip":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_Trip()
    
    ## No Use
    if net_str=="dualstream_v1_transformer":
        from model.Dualstream_v1.decoder import dualstream_tr as dualstream_v1_transformer
        return dualstream_v1_transformer()
    if net_str=="dualstream_v2_transformer":
        from model.Dualstream_v2.conv_decoder import dualstream_swin as dualstream_v2_transformer
        return dualstream_v2_transformer()
    if net_str=="dualstream_v2_resnet":
        from model.Dualstream_v2.conv_decoder import dualstream_resnet50 as dualstream_v2_resnet50
        return dualstream_v2_resnet50()
    


net = net_define(args.net)



try:
   net.load_state_dict(torch.load(args.model_dir))
   print("Checkpoint Loaded:",args.model_dir)
except:
   print("No Checkpoint Loaded.") 

model_dir = "ckpt/"+args.net+'/'
if os.path.exists(model_dir)<=0:
    os.makedirs(model_dir)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")

if args.optimizer=="Adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
elif args.optimizer=="AdamW":
    optimizer=optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
else:
    optimizer=optim.SGD(net.parameters(),lr=args.lr)

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

        inputs, labels_lung,labels_infection= data['image'], data['label_lung'], data['label_infection']

        inputs = inputs.type(torch.FloatTensor)
        labels_lung = labels_lung.type(torch.FloatTensor)
        labels_infection= labels_infection.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_lung_v, labels_infection_v= Variable(inputs.cuda(), requires_grad=False), Variable(labels_lung.cuda(),requires_grad=False), Variable(labels_infection.cuda(),requires_grad=False)
        else:
            inputs_v, labels_lung_v, labels_infection_v= Variable(inputs, requires_grad=False), Variable(labels_lung, requires_grad=False), Variable(labels_infection, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        if args.net=="dualstream_v1":
             lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection= net(inputs_v)
             loss_infection = bce_ssim_loss(infection_flow4, labels_infection_v)+bce_ssim_loss(ms1_infection, labels_infection_v)*args.multi_scale_factor+bce_ssim_loss(ms2_infection, labels_infection_v)*args.multi_scale_factor
             loss_lung = bce_ssim_loss(lung_flow4, labels_lung_v)+bce_ssim_loss(ms1_lung, labels_lung_v)*args.multi_scale_factor+bce_ssim_loss(ms2_lung, labels_lung_v)*args.multi_scale_factor
             loss=loss_lung+loss_infection
             print("Lung Loss: ",loss_lung.data)
             print("Infection Loss: ",loss_infection.data)


        elif args.net=="dualstream_v2" or args.net=="dualstream_CBAM" or args.net=="dualstream_SE" or args.net=="dualstream_Trip":
             lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection= net(inputs_v)
             loss_infection = bce_ssim_loss(infection_flow4, labels_infection_v)+bce_ssim_loss(ms1_infection, labels_infection_v)*args.multi_scale_factor+bce_ssim_loss(ms2_infection, labels_infection_v)*args.multi_scale_factor+bce_ssim_loss(ms3_infection, labels_infection_v)*args.multi_scale_factor
             loss_lung = bce_ssim_loss(lung_flow4, labels_lung_v)+bce_ssim_loss(ms1_lung, labels_lung_v)*args.multi_scale_factor+bce_ssim_loss(ms2_lung, labels_lung_v)*args.multi_scale_factor+bce_ssim_loss(ms3_lung, labels_lung_v)*args.multi_scale_factor
             loss=loss_lung+loss_infection
             print("Lung Loss: ",loss_lung.data)
             print("Infection Loss: ",loss_infection.data)

        else:
             #(Attention Network)
             infection_flow4=net(inputs_v)
             loss_infection = bce_ssim_loss(infection_flow4, labels_infection_v)
             loss=loss_infection
             print("Infection Loss: ",loss_infection.data)
                 

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

        if ite_num % 3000 == 0:  # save model every 3000 iterations

            torch.save(net.state_dict(), model_dir + "Dualstream_bsi_itr_%d_train_%3f.pth" % (ite_num, running_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
