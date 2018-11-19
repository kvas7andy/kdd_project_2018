#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:37:11 2018

@author: nicholas
"""
import torch

def prepare_optim(model, opt):
    params = [ p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=opt.lr)#, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 8], gamma=0.5)
    return optimizer, scheduler