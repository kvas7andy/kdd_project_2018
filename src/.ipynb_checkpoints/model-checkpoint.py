#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:36:42 2018

@author: nicholas
"""
import torch
from torchvision import models
import torch.nn as nn

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
] + ['AlexNet', 'alexnet']



model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# this is the pre-trained model that you want to fine-tune
class Pretrained(nn.Sequential):
    def __init__(self, model_type = 'VGG16', num_output = 2, pretrained = True, path = None):
        """
        Args:
            model (string): type of model to be used.
            num_output (int): number of neurons in the last feature layer, which
            is the number of classes of your new task
            pretrained (boolean): whether to use a pre-trained model from ImageNet
        """
        super(Pretrained, self).__init__()
        self.model_type = model_type
        self.num_output = num_output
        if self.model_type == 'Alex':
            Alex = models.alexnet()
            if pretrained:
                if path is not None:
                    # load pre-trained model
                    Alex.load_state_dict(torch.load(path))
                else:
                    Alex.load_state_dict(torch.load(path))
            num_features = Alex.classifier[6].in_features
            # Remove last layer
            new_classifier = list(Alex.classifier.children())[:-1]
            # Add a new linear layer with specified number of neurons
            new_classifier.extend([nn.Linear(num_features, self.num_output)])
            # Replace the model classifier
            Alex.classifier = nn.Sequential(*new_classifier)
            self.add_module('alex', Alex)
            
        if self.model_type == 'VGG16':
            vgg16 = models.vgg16_bn()
            if pretrained:
                # load pre-trained model
                vgg16.load_state_dict(torch.load(path))
            num_features = vgg16.classifier[6].in_features
            # Remove last layer
            new_classifier = list(vgg16.classifier.children())[:-1]
            # Add a new linear layer with specified number of neurons
            new_classifier.extend([nn.Linear(num_features, self.num_output)])
            # Replace the model classifier
            vgg16.classifier = nn.Sequential(*new_classifier)
            self.add_module('vgg16', vgg16)
        else:
            raise NotImplementedError
    def get_out_feature_size(self):
        if self.model_name in ['Alex', 'VGG16']:
            return self.num_output
        else:
            raise NotImplementedError

def prepare_model(opt):
    model = Pretrained(model_type=opt.model_type, num_output=opt.num_output, pretrained=True)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model