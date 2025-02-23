from pprint import pprint
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops, remove_self_loops
from torch_geometric.nn.conv import MessagePassing
import torch
from torch import nn
import rdkit
from tqdm.auto import tqdm
import itertools
from rdkit import Chem
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
from rdkit import RDLogger
from copy import deepcopy
#from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union
from torch import Tensor
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from rdkit import Chem
import os
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
cuda=torch.device('cuda') if torch.cuda.is_available() else 'cpu'
import sascorer
#torch.set_default_dtype(torch.float64)
from models import *
from rdkit.Chem.Crippen import MolLogP
from typing import List

ATOMIC_SYMBOL = {
    'H':0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 
    'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 
    'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 
    'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 
    'Br': 34, 'Kr': 35, 'Rb': 36, 'Sr': 37, 'Y': 38, 'Zr': 39, 'Nb': 40, 'Mo': 41, 
    'Tc': 42, 'Ru': 43, 'Rh': 44, 'Pd': 45, 'Ag': 46, 'Cd': 47, 'In': 48, 'Sn': 49, 
    'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 'Ba': 55, 'La': 56, 'Ce': 57, 
    'Pr': 58, 'Nd': 59, 'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 'Dy': 65, 
    'Ho': 66, 'Er': 67, 'Tm': 68, 'Yb': 69, 'Lu': 70, 'Hf': 71, 'Ta': 72, 'W ': 73, 
    'Re': 74, 'Os': 75, 'Ir': 76, 'Pt': 77, 'Au': 78, 'Hg': 79, 'Tl': 80, 'Pb': 81, 
    'Bi': 82, 'Po': 83, 'At': 84, 'Rn': 85, 'Fr': 86, 'Ra': 87, 'Ac': 88, 'Th': 89, 
    'Pa': 90, 'U': 91, 'Np': 92, 'Pu': 93, 'Am': 94, 'Cm': 95, 'Bk': 96, 'Cf': 97, 
    'Es': 98, 'Fm': 99, 'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 'Sg': 105, 
    'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds': 109, 'Rg': 110, 'Cn': 111, 'Nh': 112, 'Fl': 113, 
    'Mc': 114, 'Lv': 115, 'Ts': 116, 'Og': 117
}

BONDTYPE= {
    'UNSPECIFIED': 0,
    'SINGLE': 1,
    'DOUBLE': 2,
    'TRIPLE': 3,
    'QUADRUPLE': 4,
    'QUINTUPLE': 5,
    'HEXTUPLE': 6,
    'ONEANDAHALF': 7,
    'TWOANDAHALF': 8,
    'THREEANDAHALF': 9,
    'FOURANDAHALF': 10,
    'FIVEANDAHALF': 11,
    'AROMATIC': 12,
    'IONIC': 13,
    'HYDROGEN': 14,
    'THREECENTER': 15,
    'DATIVEONE': 16,
    'DATIVE': 17,
    'DATIVEL': 18,
    'DATIVER': 19,
    'OTHER': 20,
    'ZERO': 21,
    'SELF':22,
}

class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return self.num_nodes
        elif key=='dest' or key=='inbound':
            return self.num_edges
        else:
            return 0
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        else:
            return 0


class QM9(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(
        self,
        root: str,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        force_reload = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf','gdb9.sdf.csv']

    @property
    def processed_file_names(self) -> str:
        return ['gdb9_processed.pt']

    def download(self) -> None:
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(os.path.join(self.raw_dir, '3195404'),
                      os.path.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
    
    def process(self) -> None:
        suppl = Chem.SDMolSupplier(self.raw_paths[0])
        df = pd.read_csv(self.raw_paths[1])
        data_list = list()
        for i, (mol, (_, y_row)) in enumerate(zip(tqdm(suppl), df.iterrows())):
            if isinstance(mol, rdkit.Chem.Mol):
                d=Data(mol=mol)
                for k in y_row.index:
                    if k=='mol_id': continue
                    d[k]=float(y_row[k])
                data_list.append(process_molecule_sparse(d))
        self.save(data_list, self.processed_paths[0])

def find_perpendicular_vectors(X):
    if X.shape[0]==0:
        return X.detach().clone()
    # Use a reference vector that is not collinear with most vectors
    ref_vector = torch.tensor([1.234334662475, -0.523246582678, -0.913510345611], device=X.device).repeat(X.size(0), 1)
    
    # Handle edge cases where X is parallel to ref_vector ([a, 0, 0] case)
    #parallel_mask = (X[:, 0] != 0) & (X[:, 1] == 0) & (X[:, 2] == 0)
    #ref_vector[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=X.device)

    # Compute orthogonal vectors using cross product
    U = torch.cross(X, ref_vector, dim=-1)
    V = torch.cross(X, U, dim=-1)
    U /= torch.linalg.norm(U,ord=2,dim=-1).view(-1,1)
    V /= torch.linalg.norm(V,ord=2,dim=-1).view(-1,1)
    
    return U,V

def find_matching_pairs(A, B, deg_out):
    'returns a [2,N] list L wherein L[i] = (a,b) and A[a]==B[b]'
    common_values = torch.unique(A)

    matching_pairs = []

    for value in common_values:
        # Indices in A where the value occurs
        indices_A = torch.nonzero((A == value) & (A != B) & (deg_out[A]>3), as_tuple=True)[0]
        # Indices in B where the value occurs
        indices_B = torch.nonzero((B == value) & (A != B), as_tuple=True)[0]

        # Compute the Cartesian product of indices_A and indices_B
        cartesian_product = torch.cartesian_prod(indices_A, indices_B)
        matching_pairs.append(cartesian_product)

    # Step 5: Concatenate all pairs into a single tensor
    if matching_pairs:
        Y = torch.cat(matching_pairs, dim=0)
    else:
        Y = torch.empty((0, 2), dtype=torch.long, device=A.device)  # No matches found

    return Y.T

def batch_tranform(batch):
    X=batch.atom_type
    A=batch.edge_index
    #E=batch.edge_attr
    T=batch.edge_type#E[:,0]#typ
    D=batch.D#E[:,2:5]#delta (bond vector)
    U=batch.U#E[:,5:8]
    V=batch.V#E[:,8:11]
    #b=batch.batch
    y=batch.y
    # return [2,n] list 
    Y=find_matching_pairs(A[0], A[1], degree(A[1], X.shape[0]))
    ## assert (Y==Y[:,Y[0]!=Y[1]]).all()
    # remove "flipped anchor bond"  (they exist because the graph is undirected, and they are colinear with the anchor bond)
    Y=Y[:,~(A[:,Y[1]].flip(0)==A[:,Y[0]]).all(0)]
    if Y.numel()==0:
        #dest=[12], inbound=[12, 2], ang_deltas=[12]
        batch.dest=torch.empty(size=(0,),device=Y.device).long()
        batch.inbound=torch.empty(size=(0,2),device=Y.device).long()
        batch.ang_deltas=torch.empty(size=(0,),device=Y.device)
        batch.anchor_ang=torch.empty(size=(0,),device=Y.device)
        return batch
    anchor_angles_u=torch.cross(D[Y[0]], D[Y[1]]+torch.randn_like(D[Y[1]])*1e-10, dim=-1)
    anchor_angles_v=torch.cross(anchor_angles_u, D[Y[0]], dim=-1)#https://study.com/cimages/videopreview/videopreview-full/5zsq17tjsc.jpg
    anchor_angles_w=D[Y[0]]/torch.linalg.norm(D[Y[0]],ord=2,dim=-1).view(-1,1)#torch.linalg.norm(anchor_angles_u,ord=2,dim=-1).view(-1,1)
    anchor_angles_v/=torch.linalg.norm(anchor_angles_v,ord=2,dim=-1).view(-1,1)
    x_proj = -(anchor_angles_w*D[Y[1]]).sum(-1)
    y_proj = (anchor_angles_v*D[Y[1]]).sum(-1)
    anchor_ang=1.5708-torch.atan2(x_proj, y_proj)#anchor_ang is the angle between each incoming bond of an anchor bond and the anchor bond itself
    # calculate projected x
    u=(U[Y[0]]*D[Y[1]]).sum(-1)
    # calculate projected y
    v=(V[Y[0]]*D[Y[1]]).sum(-1)
    # calculate angle (radian)
    ang=torch.atan2(u,v)#ang is the angle between each pair of bonds that is connected to an anchor bond
    # get sorted index based on the angle (this is needed because the convolution goes 'around' the anchor axis)
    sorted_idx=(Y[0]*2*torch.pi+ang).argsort()
    # helper variables
    ang_sorted=ang[sorted_idx]
    Y_sorted=Y[:,sorted_idx]
    # groups is [a,b,c...] wherein each member is an index representing the end index of each group
    # what is a group?
    # assume Y is 
    # 10 10 10 11 11 12 12
    # 15 16 17 12 13 14 15
    # which means bond#10 is coming OUT from a chiral center, which have 3 bonds connecting INTO it.
    # bond#10, in this context, for the first 3 entry in Y from the left. is an anchor bond, also called a "group"
    # however, since the graph is undirected, bond#15 might be the flipped bond, which is colinear 
    # we don't want rotate around every other bond, which means we remove the flipped anchor bond
    # in the example above, there would be 3 groups whose end-index would be 2,4
    # whenever there is G groups, there len(groups) == G-1
    # which is why we concat -1 to it, representing the end=index of the final group
    groups=Y_sorted[0].diff().nonzero()[:,0]
    groups=torch.cat([groups,torch.tensor([-1],device=groups.device)])
    # `idx` explaination:
    # idx is a [2,N] matrix containing circular-pairwise indeces for every group
    # which, continuing from previous part, idx would be
    # 0 1 2 3 4 5 6
    # 1 2 0 4 3 6 5
    try:
        idx=torch.arange(Y.shape[-1], device=Y.device).repeat(2).view(2,-1)
        idx[1]=idx[0].roll(-1)
        idx[1,groups]=(groups+1).roll(1)
    except Exception as e:
        print("idx:",idx.shape)
        print("idx=",idx)
        print("groups:",groups.shape)
        print("groups=",groups)
        print("Y:",Y.shape)
        print("Y=",Y)
        raise e
    inbound=Y_sorted[1,idx.T]
    ang_deltas=((ang_sorted[idx].T.diff() + torch.pi) % (2 * torch.pi) - torch.pi).abs()
    dest=Y_sorted[0,None].T
    batch.dest=dest.squeeze(-1)
    batch.inbound=inbound
    batch.ang_deltas=ang_deltas.squeeze(-1) 
    batch.anchor_ang=anchor_ang.squeeze(-1) 
    return batch

DEFAULT_TYPE=torch.tensor([]).dtype
def process_molecule_sparse(data: Data):
    m = data.mol
    conf: Chem.Conformer=m.GetConformer()
    atoms: List[Chem.Atom]=m.GetAtoms()
    bonds = m.GetBonds()
    Nv=len(atoms)
    Ne=len(bonds)
    atom_type=torch.zeros(Nv).long()
    P=torch.empty(Nv, 3, dtype=DEFAULT_TYPE)
    A=[]
    edge_type=[]
    logp=MolLogP(m)
    #pos=dict()
    for idx,atom in enumerate(atoms):
        atom_type[idx]=ATOMIC_SYMBOL[atom.GetSymbol()]
        #pos[idx]=conf.GetAtomPosition(idx)
        edge_type.append(BONDTYPE['SELF'])
        A.append([idx,idx])
        p=conf.GetAtomPosition(idx)
        P[idx]=torch.tensor([p.x,p.y,p.z])
    for bond in bonds:
        i=bond.GetBeginAtomIdx()
        j=bond.GetEndAtomIdx()
        typ=BONDTYPE[str(bond.GetBondType())]
        
        A.append([i,j])
        A.append([j,i])
        
        
        edge_type.append(typ)
        edge_type.append(typ)
    edge_type=torch.tensor(edge_type)
    A=torch.tensor(A).long()
        
    D = P[A[:,1]]-P[A[:,0]]
    U,V=find_perpendicular_vectors(D)
    dist=torch.linalg.norm(D,ord=2,dim=-1).view(-1,1)
    #E=torch.cat([E,dist,D,U,V],-1)
    ret=MolData(atom_type=atom_type,edge_index=A.T,edge_type=edge_type,dist=dist,D=D,U=U,V=V,logp=logp,pos=P)
    for k in data.keys():
        if k!='mol':
            ret[k]=data[k]
    return batch_tranform(ret)

def transform(data):
    #print("ayo",data)
    ret= process_molecule_sparse(data)
    #print("ret",ret)
    return ret

import py3Dmol

def show(mol, style='stick'):
    mblock = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=200, height=200)
    view.addModel(mblock, 'mol')
    view.setStyle({style:{}})
    view.zoomTo()
    view.show()

def load_model(name,cls,prefix='saves'):
    checkpoint=torch.load(os.path.join(prefix,name,'checkpoint.pth'))
    instance:nn.Module = cls.from_config(checkpoint['config'])
    instance.load_state_dict(checkpoint['model_state_dict'])
    return instance

def save_model(
    name, 
    model, 
    optimizer=None, 
    scheduler=None, 
    arch=None, 
    last_epoch=None,
    loss_record=None, 
    loss_metric=None,
    total_training_iters=None,
    last_target_name=None, 
    last_batch_size=None,
    last_dataset_name=None,
    tag_uuid=True,
    or_tag_date=True,
    allow_overwrite=False,
    prefix='saves',
):
    import inspect
    if tag_uuid:
        import uuid
        name=name+'-'+uuid.uuid4().hex[:6]
    elif or_tag_date:
        import datetime
        suffix=datetime.datetime.now().strftime("%y%m%d")
        name=name+'-'+suffix
    if os.path.exists(os.path.join(prefix,name)) and not allow_overwrite:
        raise FileExistsError("File/dir already exists")
    elif not allow_overwrite:
        os.makedirs(os.path.join(prefix,name))
    #with open(os.path.join('saves',name,"source_code.py")) as f:
    #    f.write(inspect.getsource(instance))
    checkpoint = {
        'last_epoch': last_epoch,  # current training epoch
        'loss_record': loss_record if loss_record is not None else dict(),  
        'loss_metric': loss_metric, 
        'last_target_name': last_target_name,  
        'total_training_iters': total_training_iters,  # total_training_iters
        'arch': arch, 
        'last_batch_size': last_batch_size, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': model.get_config(),  # a dict of hyperparameters
        'arch': arch if arch is not None else type(model).__name__,      # e.g., obtained via subprocess from git
        # Optionally add any additional info (loss, metrics, etc.)
    }
    
    torch.save(checkpoint, os.path.join(prefix,name,'checkpoint.pth'))

def imshow(x):
    return plt.imshow(x.detach().cpu().numpy())