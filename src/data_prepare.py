#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script prepares a dataset.
@author: Nicholas
Contact: nicholas.li@connect.ust.hk
"""

import pandas as pd
import os
from os import path

import torch
import torch.utils.data
import imageio as io
import numpy as np
from skimage import img_as_float32

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class WhatEverDataSet(torch.utils.data.Dataset):
    # dataset object which stores the image paths and landmark annotations

    def __init__(self, opt, type='train'):
        """
            txt_file (string): You can specify the pathes to the images in a txt file
        """
        self.dataset_dir = opt.dataset_dir
        self.info_txt = pd.read_csv(path.join(self.dataset_dir,
                                              opt.train_file_path if type == 'train' else opt.eval_file_path),
                                    sep=' ', header=None)

        self.transform = opt.transform # True/False
        if self.tranform:
            # Use ImageNet transformation for each dataset
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            self.tranform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.num_classes = np.max(self.info_txt.iloc[:, 1].as_matrix())
        self.class_names = np.max(self.info_txt.iloc[:, 2].as_matrix())
        #self.mean = np.array((112.0, 112.0, 112.0), dtype=np.float32)
        #self.mean /= 255.0

    def __len__(self):
        return 1000

    def __getitem__(self, idx):


        landmarks = labels.astype('float').reshape(-1, 2)
        # give zero tensors in this version

        img_name = os.path.join(self.dataset_dir,
                                self.info_txt.iloc[idx, 0])
        image = io.imread(img_name)
        # the label belongs to the th class
        label = self.info_txt.iloc[idx, 1].as_matrix()
        sample = {'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)}

        if self.transform:
            sample = self.transform(sample)
        return sample


def prepare_db(opt):

    # Use ImageNet transform

    training_set = WhatEverDataSet(opt)
    if opt.eval_file_path == '':
        return {'train':training_set}
    else:
        evaluation_set = WhatEverDataSet(opt, type='eval')
        return {'train':training_set, 'eval':evaluation_set}