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

from model.network_compareSOTA.r2u_net import AttU_Net

import pytorch_ssim
import pytorch_iou

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
	
	image_dir = '/home/htihe/MDPI/Data/Task4/MDPI_Data/Testing_set/Images/'

	
	img_name_list = glob.glob(image_dir + '*.png')
	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	def net_define(net_str):
		# 12 Network
		if net_str == "couplenetc":
			from model.network_compareSOTA.couplenet import COPLENet
			return COPLENet()
		if net_str == "miniseg":
			from model.network_compareSOTA.MiniSeg import MiniSeg
			return MiniSeg()
		if net_str == "attunet":
			from model.network_compareSOTA.r2u_net import AttU_Net
			return AttU_Net()
		if net_str == "rplfunet":
			from model.network_compareSOTA.RPLFUnet import RPLFUnet
			return RPLFUnet()
		if net_str == "rplfunet":
			from model.network_compareSOTA.RPLFUnet import RPLFUnet
			return RPLFUnet()
		if net_str == "UNet_0Plus":
			from model.network_compareSOTA.UNet_0Plus import UNet
			return UNet()
		if net_str == "UNet_2D":
			from model.network_compareSOTA.unet_2D import unet_2D
			return unet_2D()
		if net_str == "UNet_2Plus":
			from model.network_compareSOTA.UNet_2Plus import UNet_2Plus
			return UNet_2Plus()
		if net_str == "UNet_3Plus":
			from model.network_compareSOTA.UNet_3Plus import UNet_3Plus
			return UNet_3Plus()
		if net_str == "UNet_Nested":
			from model.network_compareSOTA.UNet_Nested import NestedUNet
			return NestedUNet()
		if net_str == "UNet_nonlocal":
			from model.network_compareSOTA.unet_nonlocal_2D import unet_nonlocal_2D
			return unet_nonlocal_2D()
		if net_str == "SwinUNet":
			from model.network_compareSOTA.SUNet_detail import SUNet
			net = SUNet(img_size=256, patch_size=4, in_chans=3, out_chans=1,
						embed_dim=96, depths=[8, 8, 8, 8],
						num_heads=[8, 8, 8, 8],
						window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
						drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
						norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
						use_checkpoint=False, final_upsample="Dual up-sample")  # .cuda()
			return net
		if net_str == "cenet":
			from model.network_compareSOTA.cenet import CE_Net_
			return CE_Net_()
		if net_str == "cpfnet":
			from model.network_compareSOTA.CPFNet import CPFNet
			return CPFNet()
		if net_str == "cosupervision":
			from model.network_compareSOTA.co_supervision import RPLFUnet
			return RPLFUnet()

		if net_str == "segformer":
			from model.General_network.segformer import Segformer
			return Segformer()
		if net_str == "fcn":
			from model.General_network.fcn import FCNs, VGGNet
			vgg_model = VGGNet(requires_grad=True, remove_fc=True)
			fcn_model = FCNs(pretrained_net=vgg_model, n_class=1)
			return fcn_model
		if net_str == "pspnet":
			from model.General_network.pspnet import PSPNet
			return PSPNet()
		if net_str == "deeplabv3":
			from model.General_network.deeplabv3 import DeepLab
			return DeepLab()
		if net_str == "BSNet":
			from model.Add_experiment.BSNet.BSNet_Res2Net import BSNet
			return BSNet()
		if net_str == "Anamnet":
			from model.Add_experiment.anamnet import AnamNet
			return AnamNet()


	import argparse

	parser = argparse.ArgumentParser(description='Net Define')
	parser.add_argument('--net', type=str)
	parser.add_argument('--pred_dir', type=str)
	parser.add_argument('--model_dir', type=str)

	args = parser.parse_args()

	prediction_dir = args.pred_dir
	model_dir = args.model_dir

	net = net_define(args.net)
	net.load_state_dict(torch.load(model_dir))
	net.eval()
	if os.path.exists(prediction_dir)<=0:
		os.makedirs(prediction_dir)
	if torch.cuda.is_available():
		net.cuda()

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
