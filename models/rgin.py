from pprint import pprint
from convertmol import parse_sdf_file, bond_type_dict, single_bond_stereo_dict, double_bond_stereo_dict
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops, remove_self_loops

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
            MLP(mlp_dims_node[0], mlp_dims_node[1:-1], mlp_dims_node[-1], final_activation=None, dropout_rate=dropout_rate)
            for _ in range(mlp_dims_edge[-1])
        ])
        self.edge_mlp = MLP(mlp_dims_edge[0], mlp_dims_edge[1:-1], mlp_dims_edge[-1], final_activation=None, dropout_rate=dropout_rate)
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
        #print("edge_attr:",edge_attr.shape)
        edge_attr = self.edge_mlp(edge_attr)
        if edge_weight is not None:
            edge_attr *= edge_weight
        #edge_attr *= edge_weight
        #print("edge_attr")
        size = (x.size(0), x.size(0))

        out = torch.zeros(x.size(0), self.node_dimses[-1], device=x.device)
        
        for i in range(self.edge_dimses[-1]):
            h = self.propagate(edge_index, edge_attr=edge_attr, x=x,
                               size=size, i=i)
            h2=self.node_mlp[i](h)
            out = out + h2
        return out
        
    def message(self, x_j: Tensor, edge_attr, i) -> Tensor:
        return x_j*edge_attr[:,i].view(-1,1)

        
