import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import time
import os
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
    return eval_acc

def train(model, optim, sche, db, opt, model_0):
    """
    Args:
        model (torch.nn.module): the model to be trained
        optim (torch.optim.X): torch optimizer to be used
        db (torch.utils.data.Dataset): prepared tor ch dataset object
        opt: command line input from the user
    """
    # for debug
#    outputs_A = []
#    outputs_B = []
    accuracy_history = []
    if opt.active:
        # if active learning is enabled
        # Get c_A, c_B and Sc_A2B first
        # Prepare hooker to get layer features
        # We use this 2 aggregators for the whole file, so be careful 1) to empty them properly; 2) use only them as feature_maps aggregators                
        def hook_A(module, input, output):
            outputs_A.append(output.to(torch.device("cpu")).detach().numpy().reshape(output.shape[0], -1))
    
        def hook_B(module, input, output):
            outputs_B.append(output.to(torch.device("cpu")).detach().numpy().reshape(output.shape[0], -1))
    
        if 'Alex'.lower() in opt.model_type.lower():
            handleA = model.alex.features[-1].register_forward_hook(hook_A)
            handleB = model.alex.classifier[-3].register_forward_hook(hook_B)
        elif 'VGG16'.lower() in opt.model_type.lower():
            handleA = model.vgg16.features[-1].register_forward_hook(hook_A)
            handleB = model.vgg16.classifier[-3].register_forward_hook(hook_B)

        # Get c_A, c_B, Sc_A2B

        embed_dir = path.join('../datasets/c_x_A_B', opt.model_type.lower())
        if not (path.exists(embed_dir) and path.exists(path.join(embed_dir, 'c_A.npy')) and path.exists(path.join(embed_dir, 'c_B.npy'))):
            # create the directory you want to save to
            if not path.exists(embed_dir):
                os.makedirs(embed_dir)            
            outputs_A = []
            outputs_B = []
            imagenet_loader = torch.utils.data.DataLoader(db['imagenet'], batch_size=opt.batch_size, shuffle=False)
            model.eval()   
            for batch_idx, batch in enumerate(imagenet_loader):
                data = batch['image']
                if opt.cuda:
                    data = data.cuda()
                with torch.no_grad():
                    model(data)
                del data    
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


    # Start fine-tuning (transfer learning) process! epoch is only 1
    criterion = nn.CrossEntropyLoss()
    model_0.eval()
    if opt.alternate:
        current_class = 0
    for epoch in range(1, opt.epochs + 1):
        #### Here, firstly, compute score and get active learning batch of size opt.active_batch_size
        n_samples = len(db['train'])

        # sample with replacement
        sampler = torch.utils.data.sampler.WeightedRandomSampler(np.ones(n_samples)/n_samples, n_samples)
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size = opt.active_sample_size if opt.active else opt.batch_size, shuffle = False, sampler=sampler)

        # loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.eval_batch_size, shuffle=False, num_workers=4)
        # num_eval = len(db['eval'])
        # for batch_idx, batch in enumerate(loader):


        # if opt.eval:
        #     evaluate(model, db, opt)
        #     model.train()        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 50:
                break
            data = batch['image']
            target = batch['label']
            if opt.cuda:
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()   
            if opt.active:
                if opt.alternate:
                    mask = target == current_class
                    selected_target = torch.masked_select(target, mask)
                    mask = mask.unsqueeze(1)
                    if mask.sum() == 0:
                        continue
                    selected = torch.masked_select(data.view(opt.active_sample_size, -1), mask)
                    selected = selected.view(mask.sum(), 3, 224, 224)
                    data = selected
                    target = selected_target
                    current_class = 1 - current_class
                # extract feature maps and score the sampled batch
                outputs_A = []
                outputs_B = []
                model.eval()                    
                with torch.no_grad():
                    outputs = model(data)
#                assert len(outputs_A[0]) == opt.active_sample_size
#                assert len(outputs_B[0]) == opt.active_sample_size
                x_A = outputs_A[0]
                x_B = outputs_B[0]
                alpha = F.softmax(model_0(data), 1).to(torch.device("cpu")).detach().numpy()
                with torch.no_grad():
                    p = F.softmax(model(data), 1).to(torch.device("cpu")).detach().numpy()
                t = batch_idx # temperature for decaying lamb value btw distinctiveness & uncertainty
                best_indices = np.argsort(dnu.score(opt.lamb, t, p, alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B))[::-1]
#                best_indices = np.random.permutation(opt.active_sample_size)

            #### Secondly, fine-tune train the module
            # sche.step()
            model.train()

            # erase all computed gradient
            optim.zero_grad()

            # take data with maximum score
            if opt.active:
                outputs = model(data[best_indices[:opt.active_batch_size].tolist()])
                loss = criterion(outputs, target[best_indices[:opt.active_batch_size].tolist()])
            else:
                outputs = model(data)
            #_, preds = torch.max(outputs, 1)
                loss = criterion(outputs, target)
#            if batch_idx > 10:
#                print('debug')
            # compute gradient
            loss.backward()
            #train one step
            optim.step()
            if batch_idx % opt.report_every == 0:
                if opt.active:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] Actively choosen {}\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.active_sample_size, len(db['train']),
                      100. * batch_idx / len(train_loader), batch_idx*opt.active_batch_size, loss.data.item()))                    
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.batch_size, len(db['train']),
                      100. * batch_idx / len(train_loader), loss.data.item()))

            # evaluate model if specified
            if opt.eval and batch_idx % opt.eval_every == 0:
                accuracy_history.append(evaluate(model, db, opt))
                model.train()
    accuracy_history = np.array(accuracy_history)
    np.save('./history' + 'active_' + str(opt.active) +'lambda_'+ str(opt.lamb) + '_alternate_'+ str(opt.alternate) + '.npy',accuracy_history)
    if opt.active:
        handleA.remove()
        handleB.remove()
def prepare_centres(model, imagenet, opt):
    """
    Args:
        model (torch.nn.module): the original model_0 to output final layer
        db (torch.utils.data.Dataset): imagenet dataset
        opt: command line input from the user
    """
    # set the model in the evaluation mode
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



def debug():
    # some debug code, don't call the function 
    import matplotlib.pyplot as plt
    import numpy as np
    img_idx = 0
    img = data[img_idx,:,:,:].data.cpu().numpy()
    img = img.transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for _ in range(2):
        np.expand_dims(mean, axis = 0)
        np.expand_dims(std, axis = 0)
    recoverred_img = img*std + mean
    plt.imshow(recoverred_img)
    plt.pause(0.1)    
    
    # some drawing code
    path = './history_active_lambda0.01_alternate.npy'
    his = np.load(path)
    his2 = np.load('./historyactive_Falselambda_0.01_alternate_True.npy')
    plt.plot(his)
    plt.plot(his2)
    plt.xlabel('Number of queried samples')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy during fine-tuning process')

    plt.figure()
    his1 = np.load('./history_active_lambda0.01_alternate.npy')
    his2 = np.load('./historyactive_Truelambda_0.1_alternate_True.npy')
    his3 = np.load('./historyactive_Truelambda_0.5_alternate_True.npy')
    plt.plot(his1)
    plt.plot(his2)
    plt.plot(his3)
    plt.legend(['lambda = 0.01', 'lambda = 0.1', 'lambda = 0.5'])
    plt.xlabel('Number of queried samples')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy during fine-tuning process')

    plt.figure()
    his1 = np.load('./historyactive_Truelambda_0.01_alternate_False.npy')
    his2 = np.load('./historyactive_Truelambda_0.1_alternate_False.npy')
    his3 = np.load('./historyactive_Truelambda_0.5_alternate_False.npy')
    plt.plot(his1)
    plt.plot(his2)
    plt.plot(his3)
    plt.legend(['lambda = 0.01', 'lambda = 0.1', 'lambda = 0.5'])
    plt.xlabel('Number of queried samples')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy during fine-tuning process without balancing')

#    '../datasets/imagenet/train_centres.txt'