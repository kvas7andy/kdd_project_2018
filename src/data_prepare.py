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
from PIL import Image
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
        info_txt_path = path.join(self.dataset_dir,
                                  opt.train_file_path if type == 'train' else opt.eval_file_path)
        if not path.exists(info_txt_path):
            self.prepare_info(opt)

        self.info_db = pd.read_csv(info_txt_path,
                                   sep=' ', header=None,
                                   names=['path', 'label'], dtype={'path':str, 'label': np.int64})
        self.n_dataset = len(self.info_db.index)

        self.transform = opt.transform # True/False

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if self.transform:
            # Use ImageNet transformation for each dataset
            self.transform = data_transforms['train'] if type == 'train' else data_transforms['eval']

        self.num_classes = np.max(self.info_db.iloc[:, 1].values)
        #self.class_names = np.max(self.info_db.iloc[:, 2].values)
        #self.mean = np.array((112.0, 112.0, 112.0), dtype=np.float32)
        #self.mean /= 255.0

    def prepare_info(self, opt):
        if "imagenet" in self.dataset_dir:
            pass
        elif "dogvscat" in self.dataset_dir:
            # Get pd.DataFrame of files from dataset: we use only from "train" with values
            self.labels = {'dog': 0, 'cat': 1}
            file_list = os.listdir(path.join(self.dataset_dir, "train"))
            file_list.sort()
            images_records = [(path.join("train", file), self.labels[file.split('.', 1)[0]])
                              for i, file in enumerate(file_list)]
            files_db = pd.DataFrame.from_records(images_records, columns = None)

            # save files while splitting (randomly) images with seed by 70% and 30% respectively
            n_db = len(files_db.index)
            indices = np.random.RandomState(seed=opt.seed).permutation(n_db)

            files_db.ix[indices[:int(0.7*n_db)]].to_csv(path.join(self.dataset_dir, "train.txt"),
                                                        header=False, index=False, sep=' ')
            files_db.ix[indices[int(0.7*n_db):]].to_csv(path.join(self.dataset_dir, "eval.txt"),
                                                        header=False, index=False, sep=' ')
        elif "pascal" in self.dataset_dir:
            pass

    def __len__(self):
        return self.n_dataset

    def __getitem__(self, idx):
        # give image from text file
        img_name = os.path.join(self.dataset_dir,
                                self.info_db.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        # the label belongs to the th class
        label = np.array(self.info_db.iloc[idx, 1])

        #label = np.zeros((self.num_classes,), dtype=np.int64)
        #print(label.shape)
        #label[self.info_db.iloc[idx, 1]] = 1

        sample = {'image': image,
                  'label': torch.from_numpy(label)}

        return sample

def prepare_db(opt):

    # Use ImageNet tranimagesform

    training_set = WhatEverDataSet(opt)
    if opt.eval_file_path == '':
        return {'train': training_set}
    else:
        evaluation_set = WhatEverDataSet(opt, type='eval')
        return {'train': training_set, 'eval': evaluation_set}

     