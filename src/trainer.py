import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

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
        loader = torch.utils.data.DataLoader(db['eval'], batch_size = opt.batch_size, shuffle=False, num_workers = 4)
        num_eval = len(db['eval'])
        for batch_idx, batch in enumerate(loader):
            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            prediction = model(data)
            prediction = prediction.clamp(min=1e-6,max=1) # resolve some numerical issue
            eval_loss = F.nll_loss(torch.log(prediction), target).data.item()
        eval_loss /= num_eval
        print('\nTest set: Average loss: {:.6f}.'.format(eval_loss))

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
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size = opt.batch_size, shuffle = True)
        for batch_idx, batch in enumerate(train_loader):
            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            # erase all computed gradient
            optim.zero_grad()
            prediction = model(data)
            prediction = prediction.clamp(min=1e-6,max=1) # resolve some numerical issue
            
            loss = F.nll_loss(torch.log(prediction), target)
            # compute gradient
            loss.backward()
            # update parameters in the neural decision forest
            optim.step()
            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.batch_size, len(db['train']),
                      100. * batch_idx / len(train_loader), loss.data.item()))
            # evaluate model if specified
            if opt.eval and batch_idx!= 0 and batch_idx % opt.eval_every == 0:
                evaluate(model, db, opt)
                model.train()
