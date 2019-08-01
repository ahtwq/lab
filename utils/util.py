import os
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import torch.nn.functional as F


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, num_classes):
    loss_sum = 0.0
    correct = 0
    conf_matrix = np.zeros((num_classes-1,num_classes-1))
    label_list = list(np.arange(num_classes-1))

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output.squeeze(), target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        # pred = output.data.max(1, keepdim=True)[1]
        pred = torch.clamp(output, 0, num_classes-1)
        pred = torch.round(pred)
        
        ## transform 6-class to 5-class
        pred = torch.clamp(pred-1, 0)
        target_var = torch.clamp(target_var-1, 0)
		
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()


        conf_batch = confusion_matrix(list(target_var.cpu().data.numpy()), list(pred.cpu().squeeze(1).data.numpy()), label_list)
        conf_matrix += conf_batch

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
        'conf_matrix': conf_matrix,
    }


def eval(loader, model, criterion, num_classes):
    ## transform 6-class to 5-class
    loss_sum = 0.0
    correct = 0
    conf_matrix = np.zeros((num_classes-1,num_classes-1))
    label_list = list(np.arange(num_classes-1))

    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output.squeeze(), target_var)
            loss_sum += loss.item() * input.size(0)
            # pred = output.data.max(1, keepdim=True)[1]
            pred = torch.clamp(output, 0, num_classes-1)
            pred = torch.round(pred)
            
            ## transform 6-class to 5-class
            pred = torch.clamp(pred-1, 0)
            target_var = torch.clamp(target_var-1, 0)
            
            correct += pred.eq(target_var.data.view_as(pred)).sum().item()
            
            conf_batch = confusion_matrix(list(target_var.cpu().data.numpy()),list(pred.cpu().squeeze(1).data.numpy()), label_list)
            conf_matrix += conf_batch

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
        'conf_matrix': conf_matrix,
    }

def eval_complex(loader, model, criterion, num_classes, res_output=None):

    loss_sum = 0.0
    correct = 0
    conf_matrix = np.zeros((num_classes-1,num_classes-1))
    label_list = list(np.arange(num_classes-1))
    if res_output:
        txt = open(res_output, 'w')

    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            output = model(input_var)
            loss = criterion(output, target_var)

            loss_sum += loss.item() * input.size(0)
            # pred = output.data.max(1, keepdim=True)[1]

            ## regression values average
            output_new = torch.mean(output.data, 0) #--> torch([0])
            if res_output:
                txt.write(str(output_new.item()) + '\n')

            pred = torch.clamp(output_new, 0, num_classes-1)
            pred = torch.round(pred)

            ## transform 6-class to 5-class
            pred = torch.clamp(pred-1, 0)
            target_var = torch.clamp(target_var-1, 0)

            correct += pred.eq(target_var[0:1].data.view_as(pred)).sum().item()
            
            conf_batch = confusion_matrix(list(target_var[0:1].cpu().data.numpy()),list(pred.cpu().data.numpy()), label_list)
            conf_matrix += conf_batch

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / (len(loader.dataset) / 10) * 100.0,
        'conf_matrix': conf_matrix,
    }


