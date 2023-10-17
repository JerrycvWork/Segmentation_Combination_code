import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

from model.network_compareSOTA.RPLFUnet import RPLFUnet
#from model.network_compareSOTA.couplenet import COPLENet
from model.network_compareSOTA.UNet_Nested import NestedUNet as UNet_2Plus
from model.network_compareSOTA.r2u_net import AttU_Net
from model.network_compareSOTA.UNet_Nested import UNet
from model.network_compareSOTA.unet_2D import unet_2D
from model.network_compareSOTA.UNet_3Plus import UNet_3Plus
from model.network_compareSOTA.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()


	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')







if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = '/home/htihe/ECCV2022/Data/JointSeg_v2/Filter_img/'
	prediction_dir = '/home/htihe/ECCV2022/supplementary/result/SwinTransformerSys/'
	model_dir = '/home/htihe/ECCV2022/supplementary/ckpt/SwinTransformerSys/itr_42000_train_0.066369_tar_0.066369.pth'
	
	img_name_list = glob.glob(image_dir + '*.png')
	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(224),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	print("...load BASNet...")
	net = SwinTransformerSys()
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
	
		print("inferencing:",img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		d1= net(inputs_test)
	
		# normalization
		pred = d1[:,0,:,:]
		pred = normPRED(pred)
	
		# save results to test_results folder
		save_output(img_name_list[i_test],pred,prediction_dir)
