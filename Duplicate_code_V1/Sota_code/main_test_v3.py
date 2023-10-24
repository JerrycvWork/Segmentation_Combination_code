import os

import cv2
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
from model.Teacher_Student.Network_1 import dualstream_TS


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    #im = Image.fromarray(predict_np).convert('RGB')
    img_name = image_name.split("/")[-1]
    #image = io.imread(image_name)
    #imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    #pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    #imo.save(d_dir + imidx + '.png')
    cv2.imwrite(d_dir + imidx + '.png',predict_np*255)






if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Net Define')
    parser.add_argument('--net', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ckpt', type=str)

    args = parser.parse_args()

    # --------- 1. get image path and name ---------

    #/home/htihe/Advanced_Topic/PRCV2023_Semi_Supervised_Learning/Dataset/COVID_P19_20/Fold1/image

    image_dir = '/home/htihe/ECCV2022/Data/JointSeg_v2/Filter_img/'
    prediction_dir = '/home/htihe/MDPI/code/Final_Integration/Result/dualstream_TS/'
    model_dir = r"/home/htihe/MDPI/code/Final_Integration/ckpt/Teacher_Student/Dualstream_v2_bsi_itr_366000_train_0.202656_tar_0.055097.pth"

    img_name_list = glob.glob(image_dir + '*.png')

    # --------- 2. dataloader ---------
    # 1. dataload
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(512), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    print("...load BASNet...")


    net = dualstream_TS()
    #net=dualstream_TS()
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    # net=net.cpu()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        lung_flow4,infection_flow4,ms1_lung,ms1_infection,ms2_lung,ms2_infection,ms3_lung,ms3_infection= net(inputs_test)

        # normalization
        pred = lung_flow4[:, 0, :, :]
        pred = normPRED(pred)

        print(pred.shape)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir + '/lung_region/')

        pred = infection_flow4[:, 0, :, :]
        pred_2 = normPRED(pred)
        #pred_2=torch.argmax(infection_flow4,dim=1)
        print(pred_2.shape)

        # save results to test_results folder
        print(prediction_dir + '/infection_region/')
        save_output(img_name_list[i_test], pred_2, prediction_dir + '/infection_region/')

