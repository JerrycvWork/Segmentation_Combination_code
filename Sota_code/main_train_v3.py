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

from model.Dualstream_v1.decoder import dualstream as dualstream_v1
from model.Dualstream_v2.conv_decoder import dualstream as dualstream_v2
from model.Teacher_Student.Network_1 import dualstream_TS

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

def muti_bce_loss_fusion(d0, d1, d2,d3, labels_v):

    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)

    loss = loss0 + loss1*0.1 + loss2*0.1+ loss3*0.1
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data))

    return loss0, loss

bcelogit_loss=nn.BCELoss()

def muti_bce_fusion(d0, d1, d2,d3, labels_v):

    loss0 = bcelogit_loss(d0,labels_v.float())
    loss1 = bcelogit_loss(d1,labels_v.float())
    loss2 = bcelogit_loss(d2,labels_v.float())
    loss3 = bcelogit_loss(d3,labels_v.float())

    loss = loss0 + loss1*0.1 + loss2*0.1+ loss3*0.1
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data))

    return loss0, loss




# ------- 2. set the directory of training dataset --------

#data_dir = '/home/htihe/ECCV2022/Data/JointSeg_v2/Final_Train/'
#tra_image_dir = 'Refine_image_3c_split_filter/'
#tra_label_dir = 'Refine_Lung_mask_split_filter/'
#tra_label2_dir = 'Temp_Infection_Mask_split_filter/'

data_dir = '/home/htihe/ECCV2022/Data/JointSeg_v2/Final_Train/'
tra_image_dir = 'Refine_image_3c_split_filter/'
tra_label_dir = 'Refine_Lung_mask_split_filter/'
tra_label2_dir = 'Temp_Infection_Mask_split_filter/'

image_ext = '.png'
label_ext = '.png'




epoch_num = 200
batch_size_train = 1
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
        RescaleT(512),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)





# ------- 3. define model --------
# define the net
#net = dualstream_v1()
#model_dir = "/home/htihe/MDPI/code/Final_Integration/ckpt/Data_ver1/Dualstream_v1/"
#net = dualstream_TS()


net=dualstream_TS()
model_dir = "/home/htihe/MDPI/code/Final_Integration/ckpt/Teacher_Student/"


if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer=optim.AdamW(net.parameters(), lr=0.0006, betas=(0.9, 0.999), weight_decay=0)
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
        lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection= net(inputs_v)

        try:
          loss2_lung, loss_lung = muti_bce_loss_fusion(lung_flow4, ms1_lung,ms2_lung,ms3_lung,labels_lung_v)
          loss2_infection, loss_infection = muti_bce_loss_fusion(infection_flow4, ms1_infection,ms2_infection,ms3_infection,labels_infection_v)
        except:
          print("Error Occur")
          loss2_lung, loss_lung = muti_bce_loss_fusion(lung_flow4, ms1_lung, ms2_lung, ms3_lung, labels_lung_v)
          loss_infection=1
          loss2_infection=1


        loss2=loss2_lung+loss2_infection
        loss=loss_lung+loss_infection

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % 6000 == 0:  # save model every 2000 iterations

            torch.save(net.state_dict(), model_dir + "Dualstream_v2_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
