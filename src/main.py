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

    # prepare M_0(x) model for training
    pretrained_model_0 = model.Pretrained(model_type=opt.model_type, num_output=1000, pretrained=True)
    if opt.cuda:
        pretrained_model_0 = pretrained_model_0.cuda()
    else:
        pretrained_model_0 = pretrained_model_0.cpu()
    # prepare centres
    if not os.path.exists('../datasets/imagenet/train_centers'+'_'+str(opt.model_type)+'.txt'):
        imagenet = data_prepare.ImageNetSmallData(opt, type='all')
        trainer.prepare_centres(pretrained_model_0, imagenet, opt)

    # prepare data
    db = data_prepare.prepare_db(opt)
    imagenet = data_prepare.ImageNetSmallData(opt, type='centres')

    # add imagenet dataset
    db.update({'imagenet': imagenet})

    # initialize the model
    pre_trained_model = model.prepare_model(opt)

    # configurate the optimizer
    optim, sche = optimizer.prepare_optim(pre_trained_model, opt)

    # train the model
    trainer.train(pre_trained_model, optim, sche, db, opt, model_0 = pretrained_model_0)

    # save the trained model
    utils.save_model(pre_trained_model, opt)

if __name__ == '__main__':
    main()
