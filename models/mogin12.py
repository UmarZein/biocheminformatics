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

import torch
from torch import nn


def constant(value, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

class MLP(nn.Module):
    def __init__(self, input_dim, dims, output_dim, norm=nn.LayerNorm, activation=nn.LeakyReLU, final_activation=nn.Identity, dropout_rate=0.1):
        super().__init__()
        self.dims=[input_dim]+dims+[output_dim]
        self.norm=norm
        self.do_rate=dropout_rate
        self.lins=nn.ModuleList()
        self.norms=nn.ModuleList()
        self.acts=nn.ModuleList()
        self.dropouts=nn.ModuleList()
        for i in range(len(self.dims)-1):
            self.lins.append(nn.Linear(self.dims[i],self.dims[i+1]))
            
            if i+1<len(self.dims)-1:
                self.norms.append(self.norm(self.dims[i+1]))
                self.acts.append(activation())
            else:
                self.norms.append(nn.Identity())
                self.acts.append(final_activation())
            self.dropouts.append(nn.Dropout(self.do_rate))
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.lins:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
    def forward(self, x):
        #print("====")
        #print("x:", x.shape)
        #print("batch:", batch.shape)
        for lin, norm, act, do in zip(self.lins, self.norms, self.acts, self.dropouts):
            #print("----")
            x=lin(x)
            #print("x:", x.shape,norm)
            x=norm(x)
            #print("x:", x.shape)
            x=act(x)
            #print("x:", x.shape)
            x=do(x)
            #print("x:", x.shape)
        return x
    def reset_identity(self, scale=1.0):
        for d in self.dims:
            assert d==self.dims[0], "dims must be all the same"
        for layer in self.lins:
            with torch.no_grad():
                layer.weight = torch.eye(d)*scale
                layer.bias*=0
        return self
    def reset_silent(self):
        for d in self.dims:
            assert d==self.dims[0], "dims must be all the same"
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight*=0
                    layer.bias*=0
        return self

class RadiusInteractionGraph(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        #edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight

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
        mlp_dims_router: List[int],
        aggr: str = 'sum',
        dropout_rate: float = 0.0,
        activation_function=nn.LeakyReLU,
        norm=nn.LayerNorm,
        routing_loss_alpha=0.1,
        routing_loss_beta=0.02,
        routing_loss_threshold=0.01,
    ):
        super().__init__(node_dim=0,aggr=aggr)

        self.node_dimses = mlp_dims_node
        self.edge_dimses = mlp_dims_edge
        self.router_dimses = mlp_dims_router
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.aggr=aggr
        self.dropout_rate=dropout_rate
        self.num_experts=mlp_dims_router[-1]
        self.node_mlp = nn.ModuleList([
            MLP(
                mlp_dims_node[0]+mlp_dims_edge[-1], 
                mlp_dims_node[1:-1], 
                mlp_dims_node[-1],
                norm=norm,
                activation=activation_function,
                final_activation=nn.Identity, 
                dropout_rate=dropout_rate
            )
            for _ in range(self.num_experts)
        ])
        self.edge_mlp = MLP(
            mlp_dims_edge[0], 
            mlp_dims_edge[1:-1], 
            mlp_dims_edge[-1], 
            norm=nn.BatchNorm1d,
            dropout_rate=dropout_rate,
            final_activation=nn.Identity
        )
        self.shared_node_mlp = MLP(
            mlp_dims_node[0]+mlp_dims_edge[-1], 
            mlp_dims_node[1:-1], 
            mlp_dims_node[-1],
            norm=norm,
            final_activation=nn.Identity, 
            dropout_rate=dropout_rate
        )
        self.router_mlp = MLP(
            mlp_dims_router[0], 
            mlp_dims_router[1:-1], 
            mlp_dims_router[-1], 
            dropout_rate=dropout_rate,
            norm=nn.BatchNorm1d,
            final_activation=nn.Identity
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.node_mlp:
            m.reset_parameters()
        self.shared_node_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        self.router_mlp.reset_parameters()
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'mlp_dims_node': self.node_dimses,
            'mlp_dims_edge': self.edge_dimses,
            'mlp_dims_router': self.router_dimses,
            'aggr': self.aggr,
            'dropout_rate': self.dropout_rate,
            'routing_loss_alpha': self.routing_loss_alpha,
            'routing_loss_beta': self.routing_loss_beta,
            'routing_loss_threshold': self.routing_loss_threshold,
        }
    
    def forward(self, x: Tensor,
                edge_index: Adj, edge_attr: Tensor, node_batch=None):
        if node_batch is None:
            node_batch=torch.zeros_like(x[:,0]).long()
            edge_batch=torch.zeros_like(edge_index[0]).long()
        else:
            edge_batch=node_batch[edge_index[0]]
        #edge_attr: (typ, dist), |e|
        #edge_index: 2, |e|
        #print("1")
        route = self.router_mlp(edge_attr).softmax(-1)
        #print("2")
        load_balacing_loss = load_balancing_loss(route, self.routing_loss_alpha, self.routing_loss_beta, self.routing_loss_threshold)
        #print("3")
        size = (x.size(0), x.size(0))
        #print("4")
        edge_attr = self.edge_mlp(edge_attr)
        #print("5")

        h0 = self.propagate(edge_index, edge_attr=edge_attr, x=x, w=torch.tensor(1.0,device=x.device),
                               size=size, edge_idx=0)#torch.zeros(x.size(0), self.node_dimses[-1], device=x.device)
        #print("6")
        out=self.shared_node_mlp(h0)
        #print("7")
        
        for expert_idx in range(self.num_experts):
            w = route[:, expert_idx]
            h = self.propagate(edge_index, 
                               x=x,
                               edge_attr=edge_attr, 
                               w=w,
                               size=size,)
            #print("8")
            out = out + self.node_mlp[expert_idx](h)
            #print("9")
        #print("0")
        return out,load_balacing_loss
        
    def message(self, x_j: Tensor, edge_attr,w) -> Tensor:
        return torch.concat([x_j,edge_attr],-1)*w.view(-1,1)
        
class MoGIN12(nn.Module):
    def __init__(self, 
                 node_dimses, 
                 edge_dimses, 
                 router_dimses,
                 dropout_rate=0.0,
                 cutoff=10.0,
                 max_neighbors=32,
                 activation_function=nn.ReLU,
                 norm=nn.LayerNorm,
                 aggr: str = 'sum',
                 routing_loss_alpha=0.1,
                 routing_loss_beta=0.1,
                 routing_loss_threshold=0.1,
                ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.n_convs = len(node_dimses)
        self.atom_type_emb = nn.Embedding(200, node_dimses[0][0])
        self.interaction_graph=RadiusInteractionGraph(cutoff,max_neighbors)
        for mlp_dims_node, mlp_dims_edge, mlp_dims_router in zip(node_dimses, edge_dimses, router_dimses):
            self.convs.append(MoGINConv(
                mlp_dims_node, mlp_dims_edge, mlp_dims_router, 
                activation_function=activation_function, 
                aggr=aggr, norm=norm, 
                dropout_rate=dropout_rate,
                routing_loss_alpha=routing_loss_alpha, 
                routing_loss_beta=routing_loss_beta, 
                routing_loss_threshold=routing_loss_threshold
            ))
            self.norms.append(GraphNorm(mlp_dims_node[-1]))
        self.alpha=nn.Parameter(torch.empty(self.n_convs))
        self.activation=activation_function()
        self.node_dimses=node_dimses 
        self.edge_dimses=edge_dimses 
        self.router_dimses=router_dimses 
        self.dropout_rate=dropout_rate
        self.cutoff=cutoff
        self.max_neighbors=max_neighbors
        self.aggr=aggr
        self.activation_function=activation_function
        #self.dist_norm=GraphNorm(1)
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
            'router_dimses':            self.router_dimses,
            'dropout_rate':             self.dropout_rate,
            'cutoff':                   self.cutoff,
            'max_neighbors':            self.max_neighbors,
            'aggr':                     self.aggr,
            'activation_function':      self.activation_function,
            'routing_loss_alpha':       self.routing_loss_alpha,
            'routing_loss_beta':        self.routing_loss_beta,
            'routing_loss_threshold':   self.routing_loss_threshold,
        }
    
        
    def forward(self, data):
        if data.batch is not None:
            batch=data.batch
        else:
            batch=torch.zeros_like(data.pos[:,0]).long()
        edge_index, edge_length = self.interaction_graph(data.pos, batch)
        normalized_edge_length=((edge_length-2.7554)/1.1664).unsqueeze(-1)
        h = self.atom_type_emb(data.atom_type)
        total_load_balancing_loss=0
        for i,(conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_,load_balacing_loss = conv(h, edge_index, normalized_edge_length, batch)
            total_load_balancing_loss += load_balacing_loss/self.n_convs
            if i+1<len(self.convs):
                h_=norm(h_,batch)
                h_=self.activation(h_)
            h=h+self.alpha[i]*h_
        if data.batch is not None:
            h= scatter(h,data.batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h,total_load_balancing_loss