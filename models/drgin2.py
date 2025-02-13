from pprint import pprint
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.nn.norm import GraphNorm, BatchNorm
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops, remove_self_loops
from torch_geometric.nn.models.schnet import RadiusInteractionGraph
from typing import Tuple, List, Dict, Union
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
import torch
from torch import nn
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from .mlp import MLP

class RGINConv(MessagePassing):
    def __init__(
        self,
        mlp_dims_node: List[int],
        mlp_dims_edge: List[int],
        aggr: str = 'sum',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.node_dimses = mlp_dims_node
        self.edge_dimses = mlp_dims_edge
        self.aggr=aggr
        self.dropout_rate=dropout_rate
        self.node_mlp = nn.ModuleList([
            MLP(mlp_dims_node[0], mlp_dims_node[1:-1], mlp_dims_node[-1],final_activation=None, dropout_rate=dropout_rate)
            for _ in range(mlp_dims_edge[-1])
        ])
        self.edge_mlp = MLP(mlp_dims_edge[0], mlp_dims_edge[1:-1], mlp_dims_edge[-1], dropout_rate=dropout_rate,final_activation=None)
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.node_mlp:
            m.reset_parameters()
        self.edge_mlp.reset_parameters()
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'mlp_dims_node': self.node_dimses,
            'mlp_dims_edge': self.edge_dimses,
            'aggr': self.aggr,
            'dropout_rate': self.dropout_rate,
        }
    
    def forward(self, x: Tensor,
                edge_index: Adj, edge_attr: Tensor, edge_weight=None):
        #edge_attr: (typ, dist), |e|
        #edge_index: 2, |e|
        edge_attr = self.edge_mlp(edge_attr).softmax(-1)
        if edge_weight is not None:
            edge_attr = edge_attr*edge_weight
        size = (x.size(0), x.size(0))

        out = torch.zeros(x.size(0), self.node_dimses[-1], device=x.device)
        
        for edge_idx in range(self.edge_dimses[-1]):
            h = self.propagate(edge_index, edge_attr=edge_attr, x=x,
                               size=size, edge_idx=edge_idx)
            #print("h:",h.shape)
            #print("edge_idx =",edge_idx)
            h2=self.node_mlp[edge_idx](h)
            out = out + h2
        return out
        
    def message(self, x_j: Tensor, edge_attr, edge_idx) -> Tensor:
        return x_j*edge_attr[:,edge_idx].unsqueeze(-1)
        
#distance rgin
class DRGIN2(nn.Module):
    def __init__(self, 
                 node_dimses, 
                 edge_dimses, 
                 dropout_rate=0.0,
                 cutoff=10.0,
                 max_neighbors=32,
                 aggr: str = 'sum',
                ):
        super().__init__()
        self.rgins = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.atom_type_emb = nn.Embedding(200, node_dimses[0][0])
        self.interaction_graph=RadiusInteractionGraph(cutoff,max_neighbors)
        for mlp_dims_node, mlp_dims_edge in zip(node_dimses, edge_dimses):
            self.rgins.append(RGINConv(mlp_dims_node, mlp_dims_edge, dropout_rate=dropout_rate, aggr=aggr))
            self.norms.append(GraphNorm(mlp_dims_node[-1]))
        self.node_dimses=node_dimses 
        self.edge_dimses=edge_dimses 
        self.dropout_rate=dropout_rate
        self.cutoff=cutoff
        self.max_neighbors=max_neighbors
        self.aggr=aggr
        self.dist_norm=BatchNorm(1)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'node_dimses':              self.node_dimses,
            'edge_dimses':              self.edge_dimses,
            'dropout_rate':             self.dropout_rate,
            'cutoff':                   self.cutoff,
            'max_neighbors':            self.max_neighbors,
            'aggr':                     self.aggr,
        }
    
        
    def forward(self, 
                data,
               ):
        edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)
        edge_attr=self.dist_norm(edge_weight.unsqueeze(-1))
        edge_weight = (self.cutoff-edge_weight.unsqueeze(-1))/self.cutoff
        ########################
        h = self.atom_type_emb(data.atom_type)
        ########################
        #print("edge_index:",edge_index.shape)
        #print("edge_weight:",edge_weight.shape)
        #print("edge_attr:",edge_attr.shape)
        #print("h:",h.shape)
        for i,(rgin, norm) in enumerate(zip(self.rgins, self.norms)):
            #print("h:",h.shape)
            h = rgin(h, edge_index, edge_attr,edge_weight=edge_weight)
            if i+1<len(self.rgins):
                h=norm(h,batch=data.batch)
                h=h.tanh()
        if data.batch is not None:
            h= scatter(h,data.batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h
