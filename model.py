import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

class GCN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,device,output_act,dynamic=False):
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
    self.output_act = output_act
    self.out_mask = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    self.out_mask[-n_outputs:] = 1

    




  def forward(self,inp):
    #inp = inp.view(inp.shape[0],-1)
    
    hs = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    hs[:,:self.n_inputs] = inp

    edge_weights = self.edge_weights
    edge_bias = self.edge_bias
  
    for step in range(self.n_forward_steps):

        hs = torch.matmul(hs,edge_weights)
        hs[:] = torch.add(hs[:],edge_bias)

        #rel = F.relu(hs)
        #hs = where(self.out_mask,hs,rel)
        if step != self.n_forward_steps-1:
          hs = F.tanh(hs)
        
        if self.dynamic and step != self.n_forward_steps-1:
          edge_weights = torch.add(self.edge_weights, self.change_edge_weights[step])
          edge_bias = torch.add(self.edge_bias, self.change_edge_bias[step])

    return self.output_act(hs[:,-self.n_outputs:])

class SUN(nn.Module):
  def __init__(self, n_inputs,n_outputs,n_hidden,n_forward_steps,batch_size,device,output_act,init_edges=10000):
    assert n_inputs + n_outputs < n_hidden
    super(SUN, self).__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.n_forward_steps = n_forward_steps
    self.batch_size = batch_size
    self.device=device
    indices = torch.randint(0,n_hidden,[2,init_edges],device=self.device)
    values = torch.randn(init_edges,device=self.device,requires_grad=True)
    indices,values = torch_sparse.coalesce(indices, values, m=n_hidden, n=n_hidden)

    self.edge_indices = indices
    self.edge_values = values
    self.output_act = output_act
    self.out_mask = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    self.out_mask[-n_outputs:] = 1


  def forward(self,inp):
    inp = inp.view(inp.shape[0],-1)
    hs = torch.zeros([self.batch_size,self.n_hidden],device=self.device)
    hs[:,:self.n_inputs] = inp
  
    for step in range(self.n_forward_steps):
        #print(hs.shape)
        #print(self.edge_weights.shape)
        hs = hs.to_sparse()
        di,dv = hs.indices(), hs.values()
        di,dv = torch_sparse.coalesce(di, dv, m=self.batch_size, n=self.n_hidden)      
        i,v = self.edge_indicess,self.edge_values
        si,sv = torch_sparse.spspmm(i, v, di, dv, self.batch_size, self.n_hidden, self.n_hidden)
        hs = torch.sparse_coo_tensor(si, sv, [self.batch_size, self.n_hidden]).to_dense()
        print(hs.shape)

        rel = F.relu(hs)
        hs = where(self.out_mask,hs,rel)

    return self.output_act(hs[:,-self.n_outputs:])


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

