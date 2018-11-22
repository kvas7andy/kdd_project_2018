#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of cost-effective model adaptation.
@author: Nicholas
Contact: nicholas.li@connect.ust.hk
"""
import torch
import model
import data_prepare
import trainer
import optimizer
import utils

import os


def main():
    # parse command line input
    opt = utils.parse_arg()

    # Set GPU
    opt.cuda = opt.gpuid>=0
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid)
    else:
        utils.time_str("GPU acceleration is disabled.")
        
    # prepare data
    db = data_prepare.prepare_db(opt)
    imagenet = data_prepare.ImageNetSmallData(opt, type='centres')
#    imagenet = None
#
#    # add imagenet dataset
    db.update({'imagenet': imagenet})

    # initialize the model
    pre_trained_model = model.prepare_model(opt)

    # prepare M_0(x) model, which is a fixed pre-trained model
    opt.num_output = 1000
    fixed_model = model.prepare_model(opt)
    # prepare centres
    if not os.path.exists('../datasets/imagenet/train_centres.txt'):
        imagenet = data_prepare.ImageNetSmallData(opt, type='all')
        trainer.prepare_centres(fixed_model, imagenet, opt)
        
    # configurate the optimizer
    optim, sche = optimizer.prepare_optim(pre_trained_model, opt)

    # train the model
    trainer.train(pre_trained_model, optim, sche, db, opt, model_0 = fixed_model)
#    trainer.train(pre_trained_model, optim, sche, db, opt, model_0 = None)
    # save the trained model
    if opt.save:
        utils.save_model(pre_trained_model, opt)

if __name__ == '__main__':
    main()
