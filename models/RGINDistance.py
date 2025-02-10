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
from .rgin import RGINConv

class RGIND(nn.Module):
    def __init__(self, 
                 node_dimses, 
                 edge_dimses, 
                 dropout_rate=0.0, 
                 aggr: str = 'sum',
                ):
        super().__init__()
        self.atom_type_emb = nn.Embedding(atom_emb_i, atom_emb_o)
        
        self.rgins = nn.ModuleList([
            RGINConv(mlp_dims_node, mlp_dims_edge, dropout_rate=dropout_rate, aggr=aggr)
            for mlp_dims_node, mlp_dims_edge in zip(node_dimses, edge_dimses)
        ])

        self.node_dimses=node_dimses 
        self.edge_dimses=edge_dimses 
        self.dropout_rate=dropout_rate
        self.aggr=aggr

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'node_dimses':              self.node_dimses,
            'edge_dimses':              self.edge_dimses,
            'dropout_rate':             self.dropout_rate,
            'aggr':                     self.aggr,
        }
    
        
    def forward(self, 
                data,
               ):
        
        atom_type=data.atom_type
        
        ########################
        h = self.atom_type_emb(atom_type)
        ########################
        for i,rgin in enumerate(self.rgins):
            h = rgin(h, edge_index, edge_attr)
            if i+1<len(self.rgins):
                h=h.tanh()
        if batch is not None:
            h= scatter(h,batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h#*width+offset
