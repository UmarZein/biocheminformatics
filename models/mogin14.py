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
import inspect

def constant(value, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

def auto_save_hyperparams(init_fn):
    def wrapper(self, *args, **kwargs):
        # Bind the arguments to the function signature and apply defaults
        sig = inspect.signature(init_fn)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        # Save all parameters except 'self'
        self.hparams = {
            name: value 
            for name, value in bound_args.arguments.items() 
            if name != "self"
        }
        return init_fn(self, *args, **kwargs)
    return wrapper

class MLP(nn.Module):
    @auto_save_hyperparams
    def __init__(self, 
                input_dim, 
                hidden_dims, 
                output_dim, 
                norm=nn.LayerNorm, 
                final_norm=nn.Identity, 
                activation=nn.SiLU, 
                final_activation=nn.Identity, 
                dropout_rate=0.1, 
                final_dropout_rate=0.0,
                ):
        super().__init__()
        dims=[input_dim]+hidden_dims+[output_dim]
        self.lins=nn.ModuleList()
        self.norms=nn.ModuleList()
        self.acts=nn.ModuleList()
        self.dropouts=nn.ModuleList()
        for i in range(len(dims)-1):
            self.lins.append(nn.Linear(dims[i], dims[i+1]))
            if i+1<len(dims)-1:
                self.norms.append(norm(dims[i+1]))
                self.acts.append(activation())
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.norms.append(final_norm(dims[i+1]))
                self.acts.append(final_activation())
                self.dropouts.append(nn.Dropout(final_dropout_rate))
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.lins:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
    def forward(self, x):
        for lin, norm, act, do in zip(self.lins, self.norms, self.acts, self.dropouts):
            x=lin(x)
            x=norm(x)
            x=act(x)
            x=do(x)
        return x

class RadiusInteractionGraph(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32, with_self_loops=True):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.with_self_loops=with_self_loops

    def forward(self, pos: Tensor, batch: Tensor):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        if self.with_self_loops:
            edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight

def load_balancing_loss(x,alpha,beta,threshold):
    if (alpha+beta)<=1e-10 or threshold>=1-1e-10:
        return 0
    #x: [#input, #experts]
    frac=1/x.shape[-1]
    expert_bias_loss=((x.mean(-2)**2).sum()-frac)*(1/(1-frac))
    uncertainty_loss=1-(x**2).sum(-1).mean()
    t=torch.tensor((alpha+beta)*threshold)
    unfiltered=alpha*expert_bias_loss+beta*uncertainty_loss
    return (unfiltered.maximum(t)-t)*((alpha+beta)/((alpha+beta)-t))
    
#Mixture of GinConv
class MoGINConv(MessagePassing):
    @auto_save_hyperparams
    def __init__(
        self,
        mlp_dims_node: List[int],
        mlp_dims_edge: List[int],
        mlp_dims_router: List[int],
        
        aggr: str = 'sum',
        
        node_norm=nn.LayerNorm, 
        final_node_norm=nn.Identity, 
        
        edge_norm=nn.LayerNorm,
        final_edge_norm=nn.Identity, 
        
        router_norm=nn.LayerNorm,
        
        activation=nn.SiLU, 
        final_activation=nn.Identity, 
        
        dropout_rate=0.1, 
        final_dropout_rate=0.0,
        
        routing_loss_alpha=0.1,
        routing_loss_beta=0.02,
        routing_loss_threshold=0.01,

        route_filter=0.05,
    ):
        super().__init__(node_dim=0,aggr=aggr)
        self.route_filter=route_filter
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.num_experts=mlp_dims_router[-1]
        self.node_mlp = nn.ModuleList()
        self.projector = nn.ModuleList()
        for i in range(self.num_experts):
            self.node_mlp.append(MLP(
                mlp_dims_node[0]+mlp_dims_edge[-1], 
                mlp_dims_node[1:-1], 
                mlp_dims_node[-1],
                norm=node_norm, 
                final_norm=final_node_norm, 
                activation=activation, 
                final_activation=final_activation, 
                dropout_rate=dropout_rate, 
                final_dropout_rate=final_dropout_rate,
            ))
            self.projector.append(nn.Linear(mlp_dims_node[0], mlp_dims_node[0]+mlp_dims_edge[-1]))
        self.shared_node_mlp = MLP(
            mlp_dims_node[0]+mlp_dims_edge[-1], 
            mlp_dims_node[1:-1], 
            mlp_dims_node[-1],
            norm=node_norm,
            final_norm=final_node_norm, 
            activation=activation, 
            final_activation=final_activation, 
            dropout_rate=dropout_rate, 
            final_dropout_rate=final_dropout_rate,
        )
        self.edge_mlp = MLP(
            mlp_dims_edge[0], 
            mlp_dims_edge[1:-1], 
            mlp_dims_edge[-1], 
            norm=edge_norm, 
            final_norm=nn.Identity,
            activation=activation, 
            final_activation=nn.Identity, 
            dropout_rate=dropout_rate, 
            final_dropout_rate=0,
        )
        self.router_mlp = MLP(
            mlp_dims_router[0], 
            mlp_dims_router[1:-1], 
            mlp_dims_router[-1], 
            norm=router_norm, 
            final_norm=nn.Identity, # this should always be nn.Identity because it will be softmaxed
            activation=activation, 
            final_activation=nn.Identity, # this should always be nn.Identity because it will be softmaxed
            dropout_rate=dropout_rate, 
            final_dropout_rate=0,
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
        return self.hparams
    
    def forward(self, 
                x: Tensor,
                edge_index: Adj, 
                edge_attr: Tensor):
            
        route = self.router_mlp(edge_attr).softmax(-1)
        load_balacing_loss = load_balancing_loss(
            route, 
            self.routing_loss_alpha, 
            self.routing_loss_beta, 
            self.routing_loss_threshold
        )
        size = (x.size(0), x.size(0))
        edge_attr = self.edge_mlp(edge_attr)

        h0 = self.propagate(
            edge_index, 
            edge_attr=edge_attr, 
            x=x, 
            w=torch.tensor(1.0,device=x.device),
            size=size,
        )
        out=self.shared_node_mlp(h0)
        
        for expert_idx in range(self.num_experts):
            prev=self.projector[expert_idx](x)
            w = route[:, expert_idx]
            h = self.propagate(edge_index[:,w>self.route_filter], 
                               x=x,
                               edge_attr=edge_attr[w>self.route_filter], 
                               w=w[w>self.route_filter],
                               size=size,)
            out = out + self.node_mlp[expert_idx](h+prev)
        return out, load_balacing_loss
        
    def message(self, x_j: Tensor, edge_attr, w) -> Tensor:
        return torch.concat([x_j,edge_attr],-1)*w.view(-1,1)
        
class MoGIN14(nn.Module):
    @auto_save_hyperparams
    def __init__(
                self, 
        
                node_dimses, 
                edge_dimses, 
                router_dimses,
        
                cutoff=10.0,
                max_neighbors=32,
                with_self_loops=True,
        
                aggr: str = 'sum',
                
                node_norm=nn.LayerNorm, 
                edge_norm=nn.LayerNorm,
                router_norm=nn.LayerNorm,#nn.BatchNorm1d,
                
                activation=nn.SiLU, 
                
                dropout_rate=0.1, 
                final_dropout_rate=0.0,
        
                routing_loss_alpha=0.1,
                routing_loss_beta=0.02,
                routing_loss_threshold=0.01,
        
                route_filter=0.05,
                ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.routing_loss_alpha=routing_loss_alpha
        self.routing_loss_beta=routing_loss_beta
        self.routing_loss_threshold=routing_loss_threshold
        self.n_convs = len(node_dimses)
        self.atom_type_emb = nn.Embedding(200, node_dimses[0][0])
        self.interaction_graph=RadiusInteractionGraph(cutoff,max_neighbors,with_self_loops=with_self_loops)
        assert len(node_dimses)==len(edge_dimses)==len(router_dimses)
        for i, (mlp_dims_node, mlp_dims_edge, mlp_dims_router) in enumerate(zip(node_dimses, edge_dimses, router_dimses)):
            if i+1<len(node_dimses):
                self.convs.append(MoGINConv(
                    mlp_dims_node, 
                    mlp_dims_edge, 
                    mlp_dims_router, 
                    aggr=aggr,
                    node_norm=node_norm,
                    final_node_norm=node_norm,
                    edge_norm=edge_norm,
                    final_edge_norm=edge_norm,
                    router_norm=router_norm,
                    activation=activation,
                    final_activation=activation,
                    dropout_rate=dropout_rate,
                    final_dropout_rate=dropout_rate,
                    routing_loss_alpha=routing_loss_alpha,
                    routing_loss_beta=routing_loss_beta,
                    routing_loss_threshold=routing_loss_threshold,
                    route_filter=route_filter,
                ))
            else:
                self.convs.append(MoGINConv(
                    mlp_dims_node, 
                    mlp_dims_edge, 
                    mlp_dims_router, 
                    aggr=aggr,
                    node_norm=node_norm,
                    final_node_norm=nn.Identity,
                    edge_norm=edge_norm,
                    final_edge_norm=nn.Identity,
                    router_norm=router_norm,
                    activation=activation,
                    final_activation=nn.Identity,
                    dropout_rate=dropout_rate,
                    final_dropout_rate=0,
                    routing_loss_alpha=routing_loss_alpha,
                    routing_loss_beta=routing_loss_beta,
                    routing_loss_threshold=routing_loss_threshold,
                    route_filter=route_filter,
                ))
            self.norms.append(GraphNorm(mlp_dims_node[-1]))
        self.alpha=nn.Parameter(torch.empty(self.n_convs))
        self.activation=activation()
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        constant(self.alpha,1.0)
            
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return self.hparams
    
        
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
            h_,load_balacing_loss = conv(h, edge_index, normalized_edge_length)
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