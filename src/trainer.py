import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import time

def evaluate(model, db, opt):
    """
    Args:
        model (torch.nn.module): the model to be evaluated in the current stage
        db (torch.utils.data.Dataset): prepared torch dataset object
        opt: command line input from the user
    """
    model.eval()
    with torch.no_grad():
        # set the model in the evaluation mode
        eval_loss = 0
        eval_acc = 0
        loader = torch.utils.data.DataLoader(db['eval'], batch_size = opt.eval_batch_size, shuffle=False, num_workers = 4)
        num_eval = len(db['eval'])
        for batch_idx, batch in enumerate(loader):
            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            eval_loss += F.cross_entropy(outputs, target).data.item()

            eval_acc += (preds == target).sum().data.item()
        eval_loss /= num_eval
        eval_acc /= num_eval

        print('\nTest set: Average loss: {:.6f}. Average accuracy {:.6f}'.format(
            eval_loss, eval_acc))

def train(model, optim, sche, db, opt):
    """
    Args:
        model (torch.nn.module): the model to be trained
        optim (torch.optim.X): torch optimizer to be used
        db (torch.utils.data.Dataset): prepared torch dataset object
        opt: command line input from the user
    """
    for epoch in range(1, opt.epochs + 1):
        sche.step()
        model.train()
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(db['train'], batch_size = opt.batch_size, shuffle = True)
        for batch_idx, batch in enumerate(train_loader):
            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            # erase all computed gradient
            optim.zero_grad()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            preds = preds.clamp(min=1e-6,max=1) # resolve some numerical issue

            loss = F.cross_entropy(outputs, target)
            # compute gradient
            loss.backward()
            #print("Model's state_dict:")
            #if loss.data.item() != 0:
            #   print(model.alex.features[0].weight.data)
            #for param_tensor in model.state_dict():
            #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # update parameters in the neural decision forest
            #print(prediction.data)
            optim.step()

            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.batch_size, len(db['train']),
                      100. * batch_idx / len(train_loader), loss.data.item()))
            # evaluate model if specified
            if opt.eval and batch_idx!= 0 and batch_idx % opt.eval_every == 0:
                evaluate(model, db, opt)
                model.train()
