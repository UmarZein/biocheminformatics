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
    
class ConvLayer(MessagePassing):
    @auto_save_hyperparams
    def __init__(
        self,
        mlp_dims_node: List[int],
        mlp_dims_edge: List[int],
        
        aggr: str = 'sum',
        
        node_norm=nn.LayerNorm, 
        final_node_norm=nn.Identity, 
        
        edge_norm=nn.LayerNorm,
        final_edge_norm=nn.Identity, 
        
        activation=nn.SiLU, 
        final_activation=nn.Identity, 
        
        dropout_rate=0.1, 
        final_dropout_rate=0.0,
    ):
        super().__init__(node_dim=0,aggr=aggr)
        self.node_mlp=(MLP(
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
        self.projector=(nn.Linear(mlp_dims_node[0], mlp_dims_node[0]+mlp_dims_edge[-1]))
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
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projector.weight)
        self.node_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return self.hparams
    
    def forward(self, 
                x: Tensor,
                edge_index: Adj, 
                edge_attr: Tensor):
        size = (x.size(0), x.size(0))
        edge_attr = self.edge_mlp(edge_attr)

        h0 = self.propagate(
            edge_index, 
            edge_attr=edge_attr, 
            x=x, 
            w=torch.tensor(1.0,device=x.device),
            size=size,
        )
        out=self.node_mlp(h0+self.projector(x)) ############
        return out
        
    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        return torch.concat([x_j,edge_attr],-1) ###########
        
class M304B(nn.Module):
    @auto_save_hyperparams
    def __init__(
                self, 
        
                node_dimses, 
                edge_dimses, 
        
                cutoff=10.0,
                max_neighbors=32,
                with_self_loops=True,
        
                aggr: str = 'sum',
                
                node_norm=nn.LayerNorm, 
                edge_norm=nn.LayerNorm,
                
                activation=nn.SiLU, 
                
                dropout_rate=0.1, 
                final_dropout_rate=0.0,
        
                ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_convs = len(node_dimses)
        self.atom_type_emb = nn.Embedding(200, node_dimses[0][0])
        self.interaction_graph=RadiusInteractionGraph(cutoff,max_neighbors,with_self_loops=with_self_loops)
        for i, (mlp_dims_node, mlp_dims_edge) in enumerate(zip(node_dimses, edge_dimses)):
            if i+1<len(node_dimses):
                self.convs.append(ConvLayer(
                    mlp_dims_node, 
                    mlp_dims_edge, 
                    aggr=aggr,
                    node_norm=node_norm,
                    final_node_norm=node_norm,
                    edge_norm=edge_norm,
                    final_edge_norm=edge_norm,
                    activation=activation,
                    final_activation=activation,
                    dropout_rate=dropout_rate,
                    final_dropout_rate=dropout_rate,
                ))
            else:
                self.convs.append(ConvLayer(
                    mlp_dims_node, 
                    mlp_dims_edge, 
                    aggr=aggr,
                    node_norm=node_norm,
                    final_node_norm=nn.Identity,
                    edge_norm=edge_norm,
                    final_edge_norm=nn.Identity,
                    activation=activation,
                    final_activation=nn.Identity,
                    dropout_rate=dropout_rate,
                    final_dropout_rate=0,
                ))
            self.norms.append(GraphNorm(mlp_dims_node[-1]))
        self.alpha=nn.Parameter(torch.empty(self.n_convs))
        self.activation=activation()
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        constant(self.alpha,0.8)#it seems from previously trained models that alpha tends to be 0.8 across the layers
            
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
        for i,(conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_ = conv(h, edge_index, normalized_edge_length)
            if i+1<len(self.convs):
                h_=norm(h_,batch)
                h_=self.activation(h_)
            h=h_
        if data.batch is not None:
            h= scatter(h,data.batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h