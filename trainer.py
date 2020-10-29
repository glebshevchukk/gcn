from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import GCN

import yaml
import json
import uuid

seed = 7

def run_trainer(basic_config,spec_config,loss,output_act,train_dset,test_dset,transform,dynamic):
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
 
    #basic experiment settings
    batch_size = basic_config['batch_size']
    epochs =  basic_config['epochs']
    n_hidden = basic_config['n_hidden']

    #specific experiment settings
    exp_name = spec_config['exp_name']
    input_size = spec_config['input_size']
    output_size = spec_config['output_size']
    lr = spec_config['lr']
    weight_decay = float(spec_config['weight_decay'])
    n_forward_steps = spec_config['n_forward_steps']

    #create results dict
    results = {'epoch':[], 'test_loss':[], 'test_acc':[],'lr':lr, 'batch_size': batch_size, 'forward_steps':n_forward_steps,'seed':seed}

    #cuda settings
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #dataset settings
    train_loader = torch.utils.data.DataLoader(train_dset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, **test_kwargs)

    model = GCN(input_size,output_size,n_hidden,n_forward_steps,batch_size,device,output_act,dynamic).to(device)
    if dynamic:
        optimizer = optim.Adam([model.edge_weights,model.edge_bias,model.change_edge_weights, model.change_edge_bias], lr=lr,weight_decay=weight_decay)
    else:
        optimizer = optim.Adam([model.edge_weights,model.edge_bias], lr=lr,weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, loss, optimizer, epoch)
        test_loss, test_acc = test(model, device, loss, test_loader)
        results['epoch'].append(epoch)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    random_id = str(uuid.uuid4())
    results_name = exp_name +'-'+str(lr)+'-'+str(batch_size)+'-'+str(n_forward_steps)+random_id+'.json'

    with open('results/'+exp_name+'/'+ results_name,'w') as f:
        json.dump(results,f)

def train(model, device, train_loader, loss, optimizer, epoch,log_interval = 100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if data.shape[0] != model.batch_size: continue
        optimizer.zero_grad()
        
        output = model(data)

        l = loss(output, target)
        l.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))


def test(model, device, loss, test_loader):
    model.eval()
    test_loss = 0
    skipped = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if data.shape[0] != model.batch_size: 
                skipped += data.shape[0]
                continue
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= (len(test_loader.dataset)-skipped)
    test_acc = correct / (len(test_loader.dataset)-skipped)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset)-skipped,
        100. * test_acc))
    
    return test_loss, test_acc 
        