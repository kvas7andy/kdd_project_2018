#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:36:42 2018

@author: nicholas
"""
import torch
from torchvision import models
import torch.nn as nn

# this is the pre-trained model that you want to fine-tune
class Pretrained(nn.Sequential):
    def __init__(self, model_type = 'VGG16', num_output = 1000, pretrained = True, path = None):
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
            Alex = None
            if path is None:
                # $TORCH_HOME/models
                Alex = models.alexnet(pretrained=pretrained)
            else:
                Alex = models.alexnet()
                if pretrained:
                    # load pre-trained model
                    Alex.load_state_dict(torch.load(path))
            if num_output == 1000:
                self.add_module('alex', Alex)
            else:
                num_features = Alex.classifier[6].in_features
                # Remove last layer
                new_classifier = list(Alex.classifier.children())[:-1]
                # Add a new linear layer with specified number of neurons
                new_classifier.extend([nn.Linear(num_features, self.num_output)])
                # Replace the model classifier
                Alex.classifier = nn.Sequential(*new_classifier)
                # Freeze first 2 layers, acc to the paper
                for layer_idx in [0, 3]:#range(0, 6)
                    for param in Alex.features[layer_idx].parameters():
                        param.requires_grad = False
                # Get A_layer and B_layer
                # self.layer_A = Alex.features[-1] # 4th layer from last == last layer of features
                # self.layer_B = Alex.classifier[-3] # 2 layer from last (FC layers), except ReLu and others
                self.add_module('alex', Alex)
                #As well doesn't work self.layer_A = self.alex.features[-1]
                #print("DELETE it", self.alex.features[-1] is self.layer_A)
            
        if self.model_type == 'VGG16':
            vgg16 = None
            if path is None:
                # $TORCH_HOME/models
                vgg16 = models.vgg16_bn(pretrained=pretrained)
            else:
                vgg16 = models.vgg16_bn()
                if pretrained:
                    # load pre-trained model
                    vgg16.load_state_dict(torch.load(path))
            if num_output == 1000:
                self.add_module('vgg16', vgg16)
            else:
                num_features = vgg16.classifier[6].in_features
                # Remove last layer
                new_classifier = list(vgg16.classifier.children())[:-1]
                # Add a new linear layer with specified number of neurons
                new_classifier.extend([nn.Linear(num_features, self.num_output)])
                # Replace the model classifier
                vgg16.classifier = nn.Sequential(*new_classifier)
                # Freeze first 7 layers, acc to the paper
                for layer_idx in [0, 3, 7, 10, 14, 17]:#range(20)
                    for param in vgg16.features[layer_idx].parameters():
                        param.requires_grad = False
                self.add_module('vgg16', vgg16)

    # TODO: optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

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

        #model = models.resnet18()

    return model