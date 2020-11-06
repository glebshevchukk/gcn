import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_block_sparse import BlockSparseLinear
import numpy as np
import math

class NLayerFC(nn.Module):
  def __init__(self,n_inputs,n_outputs,n_hidden,n_layers,device):
    super(NLayerFC, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_layers = n_layers
    self.device=device
    self.act = F.relu

    #calculate what the inner dim is going to be to match the same number of params
    p = np.array([n_layers,n_inputs+n_outputs,-n_hidden**2])
    roots = np.roots(p)
    root = None
    for r in roots:
      if r > 0:
        root = math.ceil(r)
    if not root:
      print("No positive inter layers found!")
      return
    
    print(f"HIDDEN DIM: {root}")
    
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(n_inputs, root))

    for _ in range(self.n_layers):
      self.layers.append(nn.Linear(root, root))
    self.layers.append(nn.Linear(root, n_outputs))
    
  def forward(self,inp):
    inp = inp.view(inp.shape[0],-1)
    for i,layer in enumerate(self.layers):
      inp = layer(inp)
      if i < self.n_layers:
        inp = self.act(inp)
    return inp

#class that updates hidden state in same manner as LSTM
#adapted from https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
class LSTMGCN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,device,dynamic=False):
    assert n_inputs + n_outputs < n_hidden
    super(LSTMGCN, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_forward_steps = n_forward_steps
    self.batch_size = batch_size
    self.device=device

    self.W = nn.Parameter(torch.Tensor(self.n_hidden, self.n_hidden* 4))
    self.U = nn.Parameter(torch.Tensor(self.n_hidden, self.n_hidden * 4))
    self.bias = nn.Parameter(torch.Tensor(self.n_hidden * 4))
    self.init_weights()

  def init_weights(self):
        stdv = 1.0 / math.sqrt(self.n_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
   
  def forward(self,inp):
    hidden_seq = []
    inp = inp.view(inp.shape[0],-1)
    HS = self.n_hidden
    
    hs,c_t = torch.zeros([self.batch_size,self.n_hidden],device=self.device),torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    hs[:,:self.n_inputs] = inp

    for step in range(self.n_forward_steps):
  
        gates = hs @ self.W + hs @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
        c_t = f_t * c_t + i_t * g_t
        hs = o_t * torch.tanh(c_t)
        hidden_seq.append(hs.unsqueeze(0))
    
    hidden_seq = torch.cat(hidden_seq, dim=0)
    # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
    hidden_seq = hidden_seq.transpose(0, 1).contiguous()

    return hidden_seq[:,-1,-self.n_outputs:]



class GCN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,device,dynamic=False):
    assert n_inputs + n_outputs < n_hidden
    super(GCN, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_forward_steps = n_forward_steps
    self.batch_size = batch_size
    self.device=device
    self.edge_weights = torch.randn([self.n_hidden,self.n_hidden],requires_grad=True,device=self.device)
    self.edge_bias = torch.randn([self.n_hidden],requires_grad=True,device=self.device)
    self.dynamic = dynamic
    if self.dynamic:
      self.change_edge_weights = torch.randn([self.n_forward_steps-1, self.n_hidden,self.n_hidden],requires_grad=True,device=self.device)
      self.change_edge_bias = torch.randn([self.n_forward_steps-1, self.n_hidden],requires_grad=True,device=self.device)
  
    self.out_mask = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    self.out_mask[-n_outputs:] = 1

    #self.conv1 = nn.Conv2d(3, 6, 5)
    #self.pool = nn.MaxPool2d(2, 2)
    #self.conv2 = nn.Conv2d(6, 16, 5)


  def forward(self,inp):
    inp = inp.view(inp.shape[0],-1)

    #inp = self.pool(F.relu(self.conv1(inp)))
    #inp = self.pool(F.relu(self.conv2(inp)))
    #inp = inp.view(-1, 16 * 5 * 5)
    
    hs = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    #hs[:,:16*5*5] = inp
    hs[:,:self.n_inputs] = inp

    edge_weights = self.edge_weights
    edge_bias = self.edge_bias
  
    for step in range(self.n_forward_steps):

        hs = torch.matmul(hs,edge_weights)
        hs = torch.add(hs,edge_bias)

        if step != self.n_forward_steps-1:
          hs = F.sigmoid(hs)
        
        if self.dynamic and step != self.n_forward_steps-1:
          edge_weights = torch.add(self.edge_weights, self.change_edge_weights[step])
          edge_bias = torch.add(self.edge_bias, self.change_edge_bias[step])

    return hs[:,-self.n_outputs:]

#class for looking further into gradients
class EXP_GCN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,device,dynamic=False):
    assert n_inputs + n_outputs < n_hidden
    super(EXP_GCN, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_forward_steps = n_forward_steps
    self.batch_size = batch_size
    self.device=device
    self.edge_weights = torch.randn([self.n_hidden,self.n_hidden]).unsqueeze(0).repeat(self.n_forward_steps,1,1)
    self.edge_weights=self.edge_weights.to(device)
    self.edge_weights.requires_grad_(True)
    self.edge_bias = torch.randn([self.n_hidden],requires_grad=True,device=self.device)


  def forward(self,inp):
    inp = inp.view(inp.shape[0],-1) 
    hs = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    hs[:,:self.n_inputs] = inp

  
    for step in range(self.n_forward_steps):
        
        hs = torch.matmul(hs,self.edge_weights[step,:,:])
        hs = torch.add(hs,self.edge_bias)

        if step != self.n_forward_steps-1:
          hs = F.sigmoid(hs)

    return hs[:,-self.n_outputs:]

class SGCN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,\
    device,output_act,dynamic=False):
    assert n_inputs + n_outputs < n_hidden
    super(SGCN, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_forward_steps = n_forward_steps
    self.batch_size = batch_size
    self.device=device
    init_edges=100000
    self.fc = BlockSparseLinear(n_hidden, n_hidden, density=0.1)
    self.output_act = output_act


  def forward(self,inp):
    inp = inp.view(inp.shape[0],-1,1)
    hs = torch.zeros([self.batch_size,self.n_hidden,1],device=self.device)
    hs[:,:self.n_inputs] = inp
    print(torch.sparse.mm(self.edge_weights,hs[0]).max())
    for step in range(self.n_forward_steps):
      hs = self.fc(hs)
      if step != self.n_forward_steps-1:
        hs = F.sigmoid(hs)
    return self.output_act(hs[:,-self.n_outputs:].squeeze(-1))


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

