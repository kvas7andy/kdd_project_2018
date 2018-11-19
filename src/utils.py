#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:30:24 2018

@author: nicholas
"""
import logging
import argparse
from torch import save
import time
import os
from os import path as path

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-active_batch_size', type=int, default=1, help="fine_tune actively: active_batch_size out of batch_size")
    parser.add_argument('-eval_batch_size', type=int, default=100)
    parser.add_argument('-dataset_dir', type=str, default='../datasets/dogvscat')
    parser.add_argument('-train_file_path', type=str, default='./train.txt')    
    parser.add_argument('-eval_file_path', type=str, default='./eval.txt')
    parser.add_argument('-num_output', type=int, default=2)
    parser.add_argument('-model_type', type=str, default='Alex')
    parser.add_argument('-pretrained_model_path', type=str, default='../model/VGG16')
    parser.add_argument('-lr', type=float, default=1e-3, help="sgd: 10, adam: 0.001")
    parser.add_argument('-lamb', type=float, default=0.9, help="value for decay btw distinctiveness & uncertainty")
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-report_every', type=int, default=5)
    # whether to apply data augmentation
    parser.add_argument('-transform', type=bool, default=True)
    # whether to perform evaluation on evaluation set during training
    parser.add_argument('-eval', type=bool, default=True)
    parser.add_argument('-eval_every', type=int, default=15)
    parser.add_argument('-save_dir', type=str, default='../model')
    parser.add_argument('-save_name', type=str, default='fine_tuned_model')
    opt = parser.parse_args()
    return opt

def get_save_dir(opt):
    save_name = path.join(opt.save_dir,  opt.model_type)
    # '__model_type_'
    if not path.exists(save_name):
        os.makedirs(save_name)
    save_name = path.join(save_name, opt.save_name + '_' + time.asctime(time.localtime(time.time())).replace(" ", "_") )
    save_name += '.t7'

    return save_name

def save_model(model, opt):
    # helper function for saving a trained model
    save_name = get_save_dir(opt)
    save(model, save_name)