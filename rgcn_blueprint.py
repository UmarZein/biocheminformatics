from torch_geometric.nn.conv import RGCNConv
from constants import *
import torch
from torch import nn
from torch_geometric.data import Data

class RGCNLayer(nn.Module):
    'https://arxiv.org/pdf/1703.06103'
    def __init__(self, in_dim, out_dim, do_rate, activation=nn.Tanh):
        super().__init__()
        self.inner = RGCNConv(in_dim, out_dim, len(BONDTYPE))
        self.do = nn.Dropout(p=do_rate)
        self.act = activation() if activation is not None else None
    def forward(self, 
                data, 
                use_old=False
                #inputs = (X,A) 
                #X: (B*, N, in_dim) ; x_v^{(l)} synonymous with h_v^{(l)}
                #A: (B*, num_relations, N, N)
               ):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        x=self.inner(x,edge_index,edge_type)
        if self.act is not None:
            data = self.act(x)
        x=self.do(x)
        return Data(x=x,edge_index=edge_index, edge_attr=edge_type)
        
class RGCN(nn.Module):
    def __init__(self, input_dim, dims, output_dim, activation=nn.Tanh, final_activation=nn.Tanh, dropout_rate=0.0):
        super().__init__()
        self.node_emb=nn.Embedding(len(ATOMIC_SYMBOL), input_dim)
        self.dims=[input_dim]+dims+[output_dim]
        self.do_rate=dropout_rate
        self.layers = nn.Sequential(
            *[
                x
                for xs in [(
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=activation, do_rate=self.do_rate),
                ) if i+1<len(self.dims)-1 else (
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=final_activation, do_rate=self.do_rate),
                ) if final_activation is not None else (
                    RGCNLayer(self.dims[i],self.dims[i+1], activation=None, do_rate=self.do_rate),
                )  for i in range(len(self.dims)-1)]
                for x in xs
            ]
        )
    def forward(self, x):
        x=x.clone()
        x.x=self.node_emb(x.x)
        return self.layers(x)