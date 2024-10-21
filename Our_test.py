import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.Dualstream_v1.decoder import dualstream as dualstream_v1
from model.Dualstream_v2.conv_decoder import dualstream as dualstream_v2


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


if __name__ == '__main__':
    def net_define(net_str):
        #Our Network
        if net_str=="dualstream_v1":
            from model.Dualstream_v1.decoder import dualstream as dualstream_v1
            return dualstream_v1()
        if net_str=="dualstream_v2":
            from model.Dualstream_v2.conv_decoder import dualstream as dualstream_v2
            return dualstream_v2()
        ## Different Attention
        if net_str=="MS_guide_attemtion":
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

        # Different Attention Network
        if net_str == "MS_guide_attemtion":
            from model.Attention_Mechanism.Network_Attention.MS_guide_attention import DAF_stack
            return DAF_stack()
        if net_str == "pranet":
            from model.Attention_Mechanism.Network_Attention.Pra_attention import PraNet
            return PraNet()
        if net_str == "pyramid":
            from model.Attention_Mechanism.Network_Attention.Pytamid_Attention import ResNet50, Classifier, PAN, \
                Mask_Classifier, Seg_total
            return Seg_total()
        if net_str == "CBAM":
            from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM, \
                dualstream_SE, dualstream_Trip
            return dualstream_CBAM()
        if net_str == "SE":
            from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM, \
                dualstream_SE, dualstream_Trip
            return dualstream_SE()
        if net_str == "Triplet":
            from model.Attention_Mechanism.Convolution_Attention.Dualstream_dif_Att import dualstream_CBAM, \
                dualstream_SE, dualstream_Trip
            return dualstream_Trip()
        
    import argparse

    parser = argparse.ArgumentParser(description='Net Define')
    parser.add_argument('--net', type=str)
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--image_dir', type=str,default="Sample_Dataset/Test/image/image/")
    
    ## Testing Parameters
    parser.add_argument('--image_size', type=int, default=256)

    args = parser.parse_args()

    image_dir = args.image_dir

    img_name_list = glob.glob(image_dir + '*.png')

    # --------- 2. dataloader ---------
    # 1. dataload
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(args.image_size), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    prediction_dir = args.pred_dir
    model_dir = args.model_dir

    net = net_define(args.net)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    if os.path.exists(prediction_dir) <= 0:
        os.makedirs(prediction_dir)
    if torch.cuda.is_available():
        net.cuda()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        if args.net=="dualstream_v1":
             lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection= net(inputs_test)
        elif args.net=="dualstream_v2" or args.net=="dualstream_CBAM" or args.net=="dualstream_SE" or args.net=="dualstream_Trip":
             lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection= net(inputs_test)
        else:
             #(Attention Network)
             infection_flow4=net(inputs_test)

        # Lung Mask Visualization
        #pred = lung_flow4[:, 0, :, :]
        #pred = normPRED(pred)

        # save results to test_results folder
        #save_output(img_name_list[i_test], pred, prediction_dir + '/lung_region/')

        pred = infection_flow4[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

