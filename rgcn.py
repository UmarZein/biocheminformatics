import torch
from torch import nn
class RGCNLayer(nn.Module):
    'https://arxiv.org/pdf/1703.06103'
    def __init__(self, in_dim, out_dim, do_rate, activation=nn.Tanh):
        super().__init__()
        self.num_relations=len(BONDS)-1
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.do_rate=do_rate
        self.activation_function=activation()
        self.wr=nn.Parameter(torch.empty(self.num_relations, in_dim, out_dim))
        self.w0=nn.Parameter(torch.empty(in_dim, out_dim))
        self.init_weights()
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.wr)
        torch.nn.init.xavier_uniform_(self.w0)
    def forward(self, 
                inputs, 
                use_old=False
                #inputs = (X,A) 
                #X: (B*, N, in_dim) ; x_v^{(l)} synonymous with h_v^{(l)}
                #A: (B*, num_relations, N, N)
               ):
        if use_old:
            raise Exception("use_old is not allowed")
        X,A=inputs
        # h_i^{l+1} = act(left_side^{l} + right_side^{l})
        # see eq. (2) in the paper
        # we decide to use neither basis-decomposition nor block-diagonal-decomposition
        # because molecules can have at most 4 edge types, including disonnected edge type
        right_side=X@self.w0

        tmp1 = torch.einsum(X, [..., 0, 1], self.wr, [2, 1, 3], [..., 0, 2, 3]).transpose(-3,-2)
        #we use A[...,1:,:,:] <- the 1: is important because A[...,0,:,:] is the ZERO BOND therefor they shouldn't propagate
        tmp2 = (A[...,1:,:,:]@tmp1)
        cir = (A[...,1:,:,:].sum(-1).unsqueeze(-1))
        #print(cir.shape, tmp2.shape)
        #raise Exception("Bruh")
        # Method 1 (error in back propagation)
        #tmp2_div_cir = torch.where((tmp2 == 0) & (cir == 0), torch.tensor(0.0), tmp2 / cir)
        
        # Method 2 (bad feeling about this, because of epsilon)
        #tmp2_div_cir = tmp2 / (cir+1e-9)
        
        # Method 3
        tmp2_div_cir = tmp2*torch.where(cir==0.0, 0.0, 1/cir)
        
        
        left_side = tmp2_div_cir.sum(-3)
        h = left_side+right_side
        if self.activation_function is not None:
            h =self.activation_function(h)
        h=nn.functional.dropout(h,p=self.do_rate)
        return h,A
    def forward_old(self, inputs):
        X,A=inputs
        #X: (B*, N, in_dim) 
        #A: (B*, num_relations, N, N)

        
        # h_i^{l+1} = left_side^{l} + right_side^{l}
        # see eq. (2) in the paper
        # we decide to use neither basis-decomposition nor block-diagonal-decomposition
        # because molecules can have at most 4 edge types, including disonnected edge type
        right_side=X@self.w0

        # GCN: X@A@W
        
        left_side = torch.zeros_like(right_side)
        # A: (B*, num_relations, N, N)
        # A.sum(-1): (B*, num_relations, N)
        # X: (B*, N, in_dim) 
        # W: (num_relations, in_dim, out_dim)
        # tmp1: (B*, N, num_relations, out_dim)
        for i in range(1,A.shape[-3]):
            A_ = A[...,i,:,:]
            cir = A_.sum(-1).unsqueeze(-1) # either sum(-1) or sum(-2)
            tmp1=(X@self.wr[i])
            tmp2=A_@tmp1
            left_side += tmp2*torch.where(cir==0.0, 0.0, 1/cir)

        new_x=nn.functional.dropout(self.activation_function(left_side+right_side),p=self.do_rate)
        
        return new_x,A

class RGCN(nn.Module):
    def __init__(self, input_dim, dims, output_dim, activation=nn.Tanh, final_activation=nn.Tanh, dropout_rate=0.1):
        super().__init__()
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
        return self.layers(x)
