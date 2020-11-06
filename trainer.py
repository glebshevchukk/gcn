from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import GCN, SGCN, NLayerFC, EXP_GCN, LSTMGCN

import numpy as np
import yaml
import json
import uuid
import torchvision.models as models


def run_trainer(basic_config,spec_config,loss,train_dset,test_dset,dynamic,use_dense,n_repeat,hidden_var,n_passes):
    seed = n_repeat
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
 
    #basic experiment settings
    batch_size = basic_config['batch_size']
    epochs =  basic_config['epochs']
    n_hidden = basic_config['n_hidden'][hidden_var]

    #specific experiment settings
    exp_name = spec_config['exp_name']
    input_size = spec_config['input_size']
    output_size = spec_config['output_size']
    lr = spec_config['lr']
    weight_decay = float(spec_config['weight_decay'])
    n_forward_steps = n_passes

    #create results dict
    results = {'epoch':[], 'test_loss':[], 'test_acc':[],'lr':lr, 'batch_size': batch_size, 'forward_steps':n_forward_steps,'seed':seed,\
        'n_hidden':n_hidden,'epochs':epochs, 'gradient_mean':[]}

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
    if use_dense:
        #model = models.resnet18().to(device)
        n_layers = 3
        model = NLayerFC(input_size,output_size,n_hidden,n_layers, device).to(device)
        results['n_layers'] = n_layers
    
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

        #keep track of how many trainable params we have
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_params])
        print(f"TOTAL PARAMS: {total_params}")
        results['total_parameters'] = int(total_params)

    else:
        #model = EXP_GCN(input_size,output_size,n_hidden,n_forward_steps,batch_size,device,dynamic).to(device)
        model = LSTMGCN(input_size,output_size,n_hidden,n_forward_steps,batch_size,device,dynamic).to(device)
        #optimizer = optim.Adam([model.edge_weights,model.edge_bias], lr=lr,weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)


        #total_params = sum([np.prod(p.size()) for p in model.edge_weights]) + sum([np.prod(p.size()) for p in model.edge_bias])
        #print(f"TOTAL PARAMS: {total_params}")
        #results['total_parameters'] = int(total_params)

    #get initial performance with no training
    test_loss, test_acc = test(model, device, loss, test_loader,use_dense)
    results['epoch'].append(0)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    for epoch in range(1, epochs+1):
        gradient_means = train(model, device, train_loader, loss, optimizer, epoch,use_dense)
        #results['gradient_mean'] = gradient_means
        test_loss, test_acc = test(model, device, loss, test_loader,use_dense)
        results['epoch'].append(epoch)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    #save results
    random_id = str(uuid.uuid4())
    results_name = exp_name+'-'+str(seed)+'-'+str(lr)+'-'+str(batch_size)+'-'+str(n_forward_steps)+'-'+str(n_hidden)+'-'+str(use_dense)+'-'+random_id+'.json'
    model_path = results_name.replace('.json','.pth')
    full_res_path = 'results/'+exp_name+'/'+ results_name
    full_model_path = 'results/'+exp_name+'/'+ model_path
    with open(full_res_path,'w') as f:
        json.dump(results,f)
    
    #save model to path
    torch.save(model.state_dict(), full_model_path)


def train(model, device, train_loader, loss, optimizer, epoch,use_dense,log_interval = 100):
    model.train()
    gradient_means = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not use_dense and data.shape[0] != model.batch_size: continue
        optimizer.zero_grad()
        
        output = model(data)
        l = loss(output, target)
        l.backward()
        #gradient_means.append(model.edge_weights.grad.view(model.n_forward_steps,-1).abs().mean(1))
        
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))
    #gradient_means = torch.stack(gradient_means)
    #gradient_means = list(gradient_means.mean(0).cpu().numpy())
    #gradient_means = [str(i) for i in gradient_means]
    #print(gradient_means)
    #return gradient_means
    return None

def test(model, device, loss, test_loader,use_dense):
    model.eval()
    test_loss = 0
    skipped = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if not use_dense and data.shape[0] != model.batch_size: 
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
        