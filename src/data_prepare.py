#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script prepares a dataset.
@author: Nicholas
Contact: nicholas.li@connect.ust.hk
"""
import torch
import torch.utils.data
import imageio as io
import numpy as np
from skimage import img_as_float32

class WhatEverDataSet(torch.utils.data.Dataset):
    # dataset object which stores the image paths and landmark annotations

    def __init__(self, opt):
        """
            txt_file (string): You can specify the pathes to the images in a txt file
        """
        self.transform = opt.transform
        self.mean = np.array((112.0, 112.0, 112.0), dtype=np.float32)
        self.mean /= 255.0

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # give zero tensors in this version
        if self.transform:
            raise NotImplementedError
        image = np.ones((3,224,224), dtype=np.float32)
        # the label belongs to the th class
        label = np.array(1, dtype=np.int64)
        sample = {'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)}
        return sample


def prepare_db(opt):
    training_set = WhatEverDataSet(opt)
    if opt.eval_file_path == '':
        return {'train':training_set}
    else:
        evaluation_set = WhatEverDataSet(opt)
        return {'train':training_set, 'eval':evaluation_set}