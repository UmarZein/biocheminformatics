import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, dims, output_dim, activation=nn.Tanh, final_activation=nn.Tanh, dropout_rate=0.1):
        super().__init__()
        self.dims=[input_dim]+dims+[output_dim]
        self.do_rate=dropout_rate
        self.layers = nn.Sequential(
            *[
                x
                for xs in [(
                    nn.Linear(self.dims[i],self.dims[i+1]),
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
    def forward(self, x):
        return self.layers(x)