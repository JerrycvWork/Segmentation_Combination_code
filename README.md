# Segmentation_Combination_code

## 0. Preface

- This repository provides code for "_**Co-ERA-Net: Co-Supervision and Enhanced Region Attention for Accurate Segmentation in COVID-19 Chest Infection Images**_". 

### 0.1. Table of Contents

- [Co-ERA-Net: Co-Supervision and Enhanced Region Attention for Accurate Segmentation in COVID-19 Chest Infection Images](#)
  - [0. Preface](#)
    - [0.1. Table of Contents](#)
  - [1. Introduction](#)
    - [1.1. Task Descriptions](#)
  - [2. Proposed Methods](#)
    - [2.1. Co-ERA-Net](#)
      - [2.1.1 Overview](#)
      - [2.1.2 Data Preparation](#)
      - [2.1.3 Usage](#)
    - [2.2. Others](#)
      - [2.2.1. Overview](#)
      - [2.2.2. Usage](#)
  - [3. Another Document for Beginners](#)
  - [4. Citation](#)


## 1. Introduction

### 1.1. Task Descriptions

Pass

## 2. Proposed Methods

### 2.1. Co-ERA-Net

#### 2.1.1 Overview

Pass

#### 2.1.2 Data Preparation

The datasets provided in paper could be downloaded at [Zenodo](https://zenodo.org/records/3757476#.Xp0FhB9fgUE) and [figshare](https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488). The [Zenodo](https://zenodo.org/records/3757476#.Xp0FhB9fgUE) dataset is the training set while [figshare](https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488) dataset is the testing set.

After unzipping the data, it would be three parts of the data: images, infection masks, lung masks.

Then, place the data in the format below:

Train

--- images

--- infection masks

--- lung masks

Test

--- images

--- infection masks

--- lung masks


#### 2.1.3 Usage

After preparing the data, please run the script below for training the network:

`python Our_train.py --net "dualstream_v2"`

The parameters of the training is shown in below:

```
parser = argparse.ArgumentParser(description='Net Define')
parser.add_argument('--net', type=str, default="")
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default="train/")
parser.add_argument('--tra_image_dir', type=str, default="image/")
parser.add_argument('--tra_label_dir', type=str, default="lung_mask/")
parser.add_argument('--tra_label2_dir', type=str, default="infection_mask/")
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--multi_scale_factor', type=float, default=0.1)
```



### 2.2. Others

#### 2.2.1 Overview





#### 2.2.1 Usage







## 3. Another Document for Beginners





## 4. Citation

Please cite our paper if you find the work useful: 

@article{He2023CoERANetCA,
  title={Co-ERA-Net: Co-Supervision and Enhanced Region Attention for Accurate Segmentation in COVID-19 Chest Infection Images},
  author={Zebang He and Alex Ngai Nick Wong and Jung Sun Yoo},
  journal={Bioengineering},
  year={2023},
  volume={10},
  url={https://api.semanticscholar.org/CorpusID:260659405}
}




