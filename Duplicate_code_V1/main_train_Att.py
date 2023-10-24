import os

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

import pytorch_ssim
import pytorch_iou

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('/home/htihe/MDPI/code/Final_Integration/runs/original_model')


def iou_value(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

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






# ------- 2. set the directory of training dataset --------

#data_dir = '/home/htihe/ECCV2022/Data/JointSeg_v2/Final_Train/'
#tra_image_dir = 'Refine_image_3c_split_filter/'
#tra_label_dir = 'Refine_Lung_mask_split_filter/'
#tra_label2_dir = 'Temp_Infection_Mask_split_filter/'

#data_dir = '/home/htihe/MDPI/Data/Task4/MDPI_Data/Training_set/Set2_Filter_Lung_mask/'
#tra_image_dir = 'Images/'
#tra_label_dir = 'Lung_mask/'
#tra_label2_dir = 'Infection_mask/'

data_dir = '/home/htihe/HTI_Seminar2_XRay_COVID_Segmentation/archive/Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/'
tra_image_dir = 'images/'
tra_label_dir = 'lung masks/'
tra_label2_dir = 'infection masks/'

image_ext = '.png'
label_ext = '.png'




epoch_num = 200
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
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)


test_img_name_list = glob.glob(r"/home/htihe/MDPI/Data/Task4/MDPI_Data/Testing_set/total_test/Filter_img/" +'*' + image_ext)

test_lbl_name_list = []
test_lbl2_name_list = []
for img_path in test_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	test_lbl_name_list.append(r"/home/htihe/MDPI/Data/Task4/MDPI_Data/Testing_set/total_test/Filter_img/" + imidx + label_ext)

for img_path in test_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	test_lbl2_name_list.append(r"/home/htihe/MDPI/Data/Task4/MDPI_Data/Testing_set/total_test/Filter_img/"+ imidx + label_ext)

print("---")
print("test images: ", len(test_img_name_list))
print("test lung labels: ", len(test_lbl_name_list))
print("test infection labels: ", len(test_lbl2_name_list))
print("---")

test_num = len(tra_img_name_list)

test_salobj_dataset = SalObjDataset(
    img_name_list=test_img_name_list,
    lbl_name_list_lung=test_lbl_name_list,
    lbl_name_list_infection=test_lbl2_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=True, num_workers=2)


def net_define(net_str):
    #12 Network
    if net_str=="MS_guide_attemtion":
        from model.Attention_Mechanism.Network_Attention.MS_guide_attention import DAF_stack
        return DAF_stack()
    if net_str=="pranet":
        from model.Attention_Mechanism.Network_Attention.Pra_attention import PraNet
        return PraNet()
    if net_str == "pyramid":
        from model.Attention_Mechanism.Network_Attention.Pytamid_Attention import ResNet50, Classifier, PAN, \
            Mask_Classifier,Seg_total
        return Seg_total()
    if net_str=="CBAM":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_CBAM()
    if net_str=="SE":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_SE()
    if net_str=="Triplet":
        from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM,dualstream_SE,dualstream_Trip
        return dualstream_Trip()



import argparse

parser = argparse.ArgumentParser(description='Net Define')
parser.add_argument('--net', type=str)
parser.add_argument('--model_dir', type=str)

args = parser.parse_args()



net = net_define(args.net)
#net.load_state_dict(torch.load(args.model_dir))
model_dir = "/home/htihe/MDPI/code/Final_Integration/Add_Attention/"+args.net+'/'
if os.path.exists(model_dir)<=0:
    os.makedirs(model_dir)
if torch.cuda.is_available():
    net = net.cuda()



from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("/home/htihe/MDPI/code/Final_Integration/Add_Attention/"+args.net+'/')

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer=optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
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

        loss2_lung, loss_lung = muti_bce_loss_fusion(lung_flow4, ms1_lung,ms2_lung,ms3_lung,labels_lung_v)
        loss2_infection, loss_infection = muti_bce_loss_fusion(infection_flow4, ms1_infection,ms2_infection,ms3_infection,labels_infection_v)

        loss2=loss2_lung+loss2_infection
        loss=loss_lung+loss_infection

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    if 1:  # save model every 2000 iterations

        #torch.save(net.state_dict(), model_dir + "Dualstream_v2_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

        writer.add_scalar('training loss',loss.data,epoch)
        writer.add_scalar('lung region loss', loss_lung, epoch)
        writer.add_scalar('Infection loss', loss_infection, epoch)


    if epoch%5==0:  # save model every 2000 iterations
        torch.save(net.state_dict(), model_dir + "Dualstream_v2_bsi_itr_%d_.pth" % (epoch))




print('-------------Congratulations! Training Done!!!-------------')
