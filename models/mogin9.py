from pprint import pprint
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.norm import GraphNorm, BatchNorm
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops, remove_self_loops, dropout_node, dropout_edge
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


class RadiusInteractionGraph(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight

def constant(value, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

def load_balancing_loss(x,alpha,beta,threshold):
    if (alpha+beta)==0:
        return 0
    #x: [#input, #experts]
    #alpha: rec. [0,1]
    #beta: rec. [0,1]
    frac=1/x.shape[-1]
    expert_bias_loss=((x.mean(-2)**2).sum()-frac)*(1/(1-frac))
    uncertainty_loss=1-(x**2).sum(-1).mean()
    t=torch.tensor((alpha+beta)*threshold)
    unfiltered=alpha*expert_bias_loss+beta*uncertainty_loss
    return (unfiltered.maximum(t)-t)*((alpha+beta)/((alpha+beta)-t))
#Mixture of GinConv
#difference between mogin8: mogin9 has node and edges dropouts
class MoGINConv(MessagePassing):
    def __init__(
        self,
        mlp_dims_node: List[int],
        mlp_dims_edge: List[int],
        aggr: str = 'sum',
        dropout_rate: float = 0.0,
        activation_function=nn.LeakyReLU,
        routing_loss_alpha=0.1,
        routing_loss_beta=0.02,
        routing_loss_threshold=0.01,
        node_dropout_rate=0.0,
        edge_dropout_rate=0.0,
    ):
        super().__init__(node_dim=0,aggr=aggr)

        self.node_dimses = mlp_dims_node
        self.edge_dimses = mlp_dims_edge
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.node_dropout_rate=node_dropout_rate
        self.edge_dropout_rate=edge_dropout_rate
        self.aggr=aggr
        self.dropout_rate=dropout_rate
        self.num_experts=mlp_dims_edge[-1]
        self.node_mlp = nn.ModuleList([
            MLP(mlp_dims_node[0], mlp_dims_node[1:-1], mlp_dims_node[-1],activation=activation_function,final_activation=None, dropout_rate=dropout_rate)
            for _ in range(self.num_experts)
        ])
        self.shared_node_mlp=MLP(mlp_dims_node[0], mlp_dims_node[1:-1], mlp_dims_node[-1],final_activation=None, dropout_rate=dropout_rate)
        self.edge_mlp = MLP(mlp_dims_edge[0], mlp_dims_edge[1:-1], mlp_dims_edge[-1], dropout_rate=dropout_rate,final_activation=None)#gating network
        self.eps0 = torch.nn.Parameter(torch.empty(1))
        self.eps = torch.nn.Parameter(torch.empty(self.num_experts))
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.node_mlp:
            m.reset_parameters()
        self.shared_node_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        self.eps0.data.fill_(0.0)
        self.eps.data.fill_(0.0)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'mlp_dims_node': self.node_dimses,
            'mlp_dims_edge': self.edge_dimses,
            'node_dropout_rate': self.node_dropout_rate,
            'edge_dropout_rate': self.edge_dropout_rate,
            'aggr': self.aggr,
            'dropout_rate': self.dropout_rate,
            'routing_loss_alpha': self.routing_loss_alpha,
            'routing_loss_beta': self.routing_loss_beta,
            'routing_loss_threshold': self.routing_loss_threshold,
        }
    
    def forward(self, x: Tensor,
                edge_index: Adj, edge_attr: Tensor, edge_weight=None):
        #edge_attr: (typ, dist), |e|
        #edge_index: 2, |e|
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_index[0]).view(-1,1)
        #_, edge_mask0 = dropout_edge(edge_index, p=self.edge_dropout_rate, training=self.training)
        #_, edge_mask1, _ = dropout_node(edge_index, p=self.node_dropout_rate, training=self.training)
        #edge_mask=edge_mask0&edge_mask1
        #print("edge_mask:",edge_mask)
        #edge_index=edge_index[:,edge_mask]
        #edge_attr=edge_attr[edge_mask]
        #edge_weight=edge_weight[edge_mask]
        edge_attr = self.edge_mlp(edge_attr).softmax(-1)
        load_balacing_loss = load_balancing_loss(edge_attr, self.routing_loss_alpha, self.routing_loss_beta, self.routing_loss_threshold)
        edge_attr = edge_attr*edge_weight
        size = (x.size(0), x.size(0))

        h0 = self.propagate(edge_index, edge_attr=edge_weight, x=x,
                               size=size, edge_idx=0)#torch.zeros(x.size(0), self.node_dimses[-1], device=x.device)
        out=self.shared_node_mlp(h0+x*(1+self.eps0))
        
        for edge_idx in range(self.edge_dimses[-1]):
            
            h = self.propagate(edge_index, edge_attr=edge_attr, x=x,
                               size=size, edge_idx=edge_idx)
            h2=self.node_mlp[edge_idx](h+x*(1+self.eps[edge_idx]))
            out = out + h2
        return out,load_balacing_loss
        
    def message(self, x_j: Tensor, edge_attr, edge_idx) -> Tensor:
        return x_j*edge_attr[:,edge_idx].unsqueeze(-1)
        
class MoGIN9(nn.Module):
    def __init__(self, 
                 node_dimses, 
                 edge_dimses, 
                 dropout_rate=0.0,
                 cutoff=10.0,
                 max_neighbors=32,
                 activation_function=nn.LeakyReLU,
                 aggr: str = 'sum',
                 routing_loss_alpha=0.1,
                 routing_loss_beta=0.1,
                 routing_loss_threshold=0.1,
                 node_dropout_rate=0.0,
                 edge_dropout_rate=0.0,
                ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.node_dropout_rate=node_dropout_rate
        self.edge_dropout_rate=edge_dropout_rate
        self.n_convs = len(node_dimses)
        self.atom_type_emb = nn.Embedding(200, node_dimses[0][0])
        self.interaction_graph=RadiusInteractionGraph(cutoff,max_neighbors)
        for mlp_dims_node, mlp_dims_edge in zip(node_dimses, edge_dimses):
            self.convs.append(MoGINConv(
                mlp_dims_node, mlp_dims_edge, aggr=aggr,
                dropout_rate=dropout_rate, node_dropout_rate=node_dropout_rate, edge_dropout_rate=edge_dropout_rate,
                routing_loss_alpha=routing_loss_alpha, routing_loss_beta=routing_loss_beta, routing_loss_threshold=routing_loss_threshold))
            self.norms.append(GraphNorm(mlp_dims_node[-1]))
        self.alpha=nn.Parameter(torch.empty(self.n_convs))
        self.activation=activation_function()
        self.node_dimses=node_dimses 
        self.edge_dimses=edge_dimses 
        self.dropout_rate=dropout_rate
        self.cutoff=cutoff
        self.max_neighbors=max_neighbors
        self.aggr=aggr
        self.activation_function=activation_function
        self.dist_norm=GraphNorm(1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        constant(self.alpha,1.0)
            
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
            'activation_function':      self.activation_function,
            'routing_loss_alpha':       self.routing_loss_alpha,
            'routing_loss_beta':        self.routing_loss_beta,
            'routing_loss_threshold':   self.routing_loss_threshold,
            'node_dropout_rate':        self.node_dropout_rate,
            'edge_dropout_rate':        self.edge_dropout_rate,
        }
    
        
    def forward(self, 
                data,
               ):
        edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)
        if data.batch is not None:
            edge_attr=self.dist_norm(edge_weight.unsqueeze(-1), data.batch[edge_index[0]])
        else:
            edge_attr=self.dist_norm(edge_weight.unsqueeze(-1), torch.zeros_like(edge_index[0]).long())
        edge_weight = ((self.cutoff)-edge_weight.unsqueeze(-1))/(self.cutoff)
        ########################
        h = self.atom_type_emb(data.atom_type)
        ########################
        #print("edge_index:",edge_index.shape)
        #print("edge_weight:",edge_weight.shape)
        #print("edge_attr:",edge_attr.shape)
        #print("h:",h.shape)
        total_load_balancing_loss=0
        for i,(conv, norm) in enumerate(zip(self.convs, self.norms)):
            #print("h:",h.shape)
            h_,load_balacing_loss = conv(h, edge_index, edge_attr, edge_weight=edge_weight)
            total_load_balancing_loss+=load_balacing_loss/self.n_convs
            #print("h_:",h_.shape)
            if i+1<len(self.convs):
                h_=norm(h_,batch=data.batch)
                h_=self.activation(h_)
            h=h+self.alpha[i]*h_
        if data.batch is not None:
            h= scatter(h,data.batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h,total_load_balancing_loss
