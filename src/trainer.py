import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import time
from os import path
import distinctiveness_and_uncertainty as dnu

def evaluate(model, db, opt):
    """
    Args:
        model (torch.nn.module): the model to be evaluated in the current stage
        db (torch.utils.data.Dataset): prepared torch dataset object
        opt: command line input from the user
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
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

            eval_loss += criterion(outputs, target).data.item()

            eval_acc += (preds == target).sum().data.item()
        eval_loss /= num_eval
        eval_acc /= num_eval

        print('\nTest set: Average loss: {:.6f}. Average accuracy {:.6f}'.format(
            eval_loss, eval_acc))

def train(model, optim, sche, db, opt, model_0):
    """
    Args:
        model (torch.nn.module): the model to be trained
        optim (torch.optim.X): torch optimizer to be used
        db (torch.utils.data.Dataset): prepared torch dataset object
        opt: command line input from the user
    """
    # Get c_A, c_B and Sc_A2B first
    # Prepare hooker to get layer features
    # We use this 2 aggregators for the whole file, so be careful 1) to empty them properly; 2) use only them as feature_maps aggregators
    outputs_A = []
    outputs_B = []

    def hook_A(module, input, output):
        outputs_A.append(output.to(torch.device("cpu")).detach().numpy().reshape(output.shape[0], -1))

    def hook_B(module, input, output):
        outputs_B.append(output.to(torch.device("cpu")).detach().numpy().reshape(output.shape[0], -1))

    if 'Alex'.lower() in opt.model_type.lower():
        model.alex.features[-1].register_forward_hook(hook_A)
        model.alex.classifier[-3].register_forward_hook(hook_B)
    elif 'VGG16'.lower() in opt.model_type.lower():
        model.vgg16.features[-1].register_forward_hook(hook_A)
        model.vgg16.classifier[-3].register_forward_hook(hook_B)

    # Get c_A, c_B, Sc_A2B
    imagenet_loader = torch.utils.data.DataLoader(db['imagenet'], batch_size=opt.batch_size, shuffle=False)
    model.eval()
    embed_dir = path.join('../datasets/c_x_A_B', opt.model_type.lower())
    if not (path.exists(embed_dir) and path.exists(path.join(embed_dir, 'c_A.npy')) and path.exists(path.join(embed_dir, 'c_B.npy'))):
        outputs_A = []
        outputs_B = []

        for batch_idx, batch in enumerate(imagenet_loader):
            data = batch['image']
            if opt.cuda:
                with torch.no_grad():
                    data = data.cuda()
            with torch.no_grad():
                model(data)

        #assert len(outputs_A) == 1000
        #assert len(outputs_B) == 1000
        c_A = outputs_A = np.vstack(outputs_A)
        c_B = outputs_B = np.vstack(outputs_B)

        np.save(path.join(embed_dir, 'c_A.npy'), c_A)
        np.save(path.join(embed_dir, 'c_B.npy'), c_B)
    else:
        c_A = np.load(path.join(embed_dir, 'c_A.npy'))
        c_B = np.load(path.join(embed_dir, 'c_B.npy'))

    if not path.exists(path.join(embed_dir, 'Sc_A2B.npy')):
        ScA = dnu.Sx_generator(c_A, c_A)
        ScB = dnu.Sx_generator(c_B, c_B)
        Sc_A2B = ScA - ScB

        np.save(path.join(embed_dir, 'Sc_A2B.npy'), Sc_A2B)
    else:
        Sc_A2B = np.load(path.join(embed_dir, 'Sc_A2B.npy'))


    # Start fine-tuning process! epoch is only 1
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, opt.epochs + 1):
        #### Here, firstly, compute score and get active learning batch of size opt.active_batch_size
        n_samples = len(db['train'])

        # sample with replacement
        sampler = torch.utils.data.sampler.WeightedRandomSampler(np.ones(n_samples)/n_samples, n_samples)
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size = opt.batch_size, shuffle = False, sampler=sampler)

        # loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.eval_batch_size, shuffle=False, num_workers=4)
        # num_eval = len(db['eval'])
        # for batch_idx, batch in enumerate(loader):


        # if opt.eval:
        #     evaluate(model, db, opt)
        #     model.train()
        model_0.eval()
        for batch_idx, batch in enumerate(train_loader):
            outputs_A = []
            outputs_B = []

            model.eval()

            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()

            idx = 0
            with torch.no_grad():
                outputs = model(data)
            assert len(outputs_A[0]) == opt.batch_size
            assert len(outputs_B[0]) == opt.batch_size
            x_A = outputs_A[0]
            x_B = outputs_B[0]
            alpha = F.softmax(model_0(data), 1).to(torch.device("cpu")).detach().numpy()
            p = F.softmax(model(data), 1).to(torch.device("cpu")).detach().numpy()
            t = batch_idx # temperature for decaying lamb value btw distinctiveness & uncertainty
            best_indices = np.argsort(dnu.score(opt.lamb, t, p, alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B))[::-1]


            #### Secondly, actively train the module
            # sche.step()
            model.train()

            # erase all computed gradient
            optim.zero_grad()

            # take data with maximum score
            outputs = model(data[best_indices[:opt.active_batch_size].tolist()])
            #_, preds = torch.max(outputs, 1)

            loss = criterion(outputs, target[best_indices[:opt.active_batch_size].tolist()])

            # compute gradient
            loss.backward()

            #train one step
            optim.step()

            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.batch_size, len(db['train']),
                      100. * batch_idx / len(train_loader), loss.data.item()))

            # evaluate model if specified
            if opt.eval and batch_idx != 0 and batch_idx % opt.eval_every == 0:
                evaluate(model, db, opt)
                model.train()

def prepare_centres(model, imagenet, opt):
    """
    Args:
        model (torch.nn.module): the original model_0 to output final layer
        db (torch.utils.data.Dataset): imagenet dataset
        opt: command line input from the user
    """
    model.eval()

    outputs_layer = []
    outputs_label = []

    def hook(module, input, output):
        outputs_layer.append(output.to(torch.device("cpu")).detach().numpy().reshape(opt.batch_size, -1))

    if 'Alex'.lower() in opt.model_type.lower():
        hook = model.alex.features[-1].register_forward_hook(hook)
    elif 'VGG16'.lower() in opt.model_type.lower():
        hook = model.vgg16.features[-1].register_forward_hook(hook)

    with torch.no_grad():
        # set the model in the evaluation mode
        loader = torch.utils.data.DataLoader(imagenet, batch_size = opt.batch_size, shuffle=False, num_workers = 4)

        for batch_idx, batch in enumerate(loader):
            data = batch['image']
            target = batch['label']
            outputs_label += batch['label'].data.tolist()
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            model(data)

        # remove hook from module
        hook.remove()
        outputs_layer = np.vstack(outputs_layer)
        outputs_label = np.array(outputs_label).reshape(-1)
        assert outputs_label.shape[0] == outputs_layer.shape[0]

    outputs_average = {}
    for label in np.sort(np.unique(outputs_label)):
        label_eq = outputs_label == label

        assert len(label_eq.shape) == 1
        outputs_average.update({label: np.mean(outputs_layer * label_eq[:, np.newaxis], axis=0)})
        assert len(outputs_average[label].shape) == 1

    outputs_min = []
    for label in np.unique(outputs_label):
        label_eq = outputs_label == label
        min_idx = np.argmin(np.sum(((outputs_layer
                 - outputs_average[label][np.newaxis, :])* label_eq[:, np.newaxis])**2, axis=-1))
        outputs_min += [min_idx]
    print(outputs_min)

    with open('../datasets/imagenet/train.txt') as f:
        content = np.array(f.readlines())
    with open('../datasets/imagenet/train_centres.txt', "w") as f:
        f.write("".join(content[outputs_min].tolist()))











    '../datasets/imagenet/train_centres.txt'