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
    # initializa the model
    pre_trained_model = model.prepare_model(opt)

    # configurate the optimizer
    optim, sche = optimizer.prepare_optim(pre_trained_model, opt)

    # train the model
    trainer.train(pre_trained_model, optim, sche, db, opt)

    # save the trained model
    utils.save_model(pre_trained_model, opt)

if __name__ == '__main__':
    main()
