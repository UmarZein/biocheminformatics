import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, dims, output_dim, activation=nn.LeakyReLU, final_activation=nn.Identity, dropout_rate=0.1):
        super().__init__()
        self.dims=[input_dim]+dims+[output_dim]
        self.do_rate=dropout_rate
        self.layers = nn.Sequential(
            *[
                x
                for xs in [(
                    nn.Linear(self.dims[i],self.dims[i+1]),
                    nn.BatchNorm1d(self.dims[i+1]),
                    activation(),
                    nn.Dropout(self.do_rate),
                ) if i+1<len(self.dims)-1 else (
                    nn.Linear(self.dims[i],self.dims[i+1]),
                    final_activation(),
                    nn.Dropout(self.do_rate),
                ) if final_activation is not None else (
                    nn.Linear(self.dims[i],self.dims[i+1]),
                    nn.Dropout(self.do_rate),
                )  for i in range(len(self.dims)-1)]
                for x in xs
            ]
        )
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
    def forward(self, x):
        return self.layers(x)
    def reset_identity(self, scale=1.0):
        for d in self.dims:
            assert d==self.dims[0], "dims must be all the same"
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
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
                