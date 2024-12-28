from pprint import pprint
from convertmol import parse_sdf_file, bond_type_dict, single_bond_stereo_dict, double_bond_stereo_dict
from torch_geometric.data import Data
from torch_geometric.nn import *
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops
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
from torch.utils.data import Dataset, DataLoader
import os
from constants import *
from rgcn_blueprint import RGCN
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
import sascorer
torch.set_default_device('cpu:0')#'cuda'

from neo4j import GraphDatabase, RoutingControl


URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")



from neo4j import Transaction, ManagedTransaction
def iterate_from_source_batched(tx: ManagedTransaction, name: str, batch_size):
    assert isinstance(batch_size, int)
    assert batch_size>0
    query = f"""
    MATCH (m:Molecule)-[:PART_OF]->(s:Source {{name: $name}})
    MATCH (a:Atom)-[:PART_OF]->(m)
    OPTIONAL MATCH (a)-[b:BOND]->(c:Atom)
    RETURN m, a, b, c
    """
    ready_mols=[]
    mol_buffer={}
    for record in tx.run(query,{"name":name}):
        #print('record')
        m = record["m"]
        a = record["a"]
        b = record.get("b",None)
        c = record.get("c",None)

        aIdx=a['Idx']
        bIdx=None
        cIdx=None
        if b:
            bIdx=b['Idx']
        if c:
            cIdx=c['Idx']
        _Name = m['_Name']

        if _Name not in mol_buffer:
            # Initialize molecule entry
            mol_buffer[_Name] = {
                "molecule": m,
                "atoms": {
                    aIdx:a,
                },
                "bonds": {}
            }
                    
            if b:
                mol_buffer[_Name]['bonds'][bIdx]=b
            if c:
                mol_buffer[_Name]['atoms'][cIdx]=c
                
        elif mol_buffer[_Name] != True: #if it's true, it means that it's already been yielded
            if aIdx not in mol_buffer[_Name]['atoms']:
                mol_buffer[_Name]['atoms'][aIdx]=a
            if bIdx and bIdx not in mol_buffer[_Name]['bonds']:
                mol_buffer[_Name]['bonds'][bIdx]=b
            if cIdx and cIdx not in mol_buffer[_Name]['atoms']:
                mol_buffer[_Name]['atoms'][cIdx]=c
        else:
            continue
        if len(mol_buffer[_Name]['atoms'])==m['NumAtoms'] and len(mol_buffer[_Name]['bonds'])==m['NumBonds']:
            #print("appending molecule",_Name,len(mol_buffer[_Name]['atoms']),len(mol_buffer[_Name]['bonds']))
            ready_mols.append(mol_buffer[_Name])
            mol_buffer[_Name]=True
            if len(ready_mols)>=batch_size:
                yield ready_mols
                ready_mols.clear()
    for k,v in mol_buffer.items():
        if v != True:
            ready_mols.append(v)
    if len(ready_mols)>0:
        yield ready_mols


from neo4j import Transaction, ManagedTransaction
def stream_from_source(tx: ManagedTransaction, name: str):
    query = f"""
    MATCH (m:Molecule)-[:PART_OF]->(s:Source {{name: $name}})
    MATCH (a:Atom)-[:PART_OF]->(m)
    OPTIONAL MATCH (a)-[b:BOND]->(c:Atom)
    RETURN m, a, b, c
    """
    mol_buffer=dict()
    for record in tx.run(query,{"name":name}):
        #print('record',record['m']['_Name'])
        m = record["m"]
        a = record["a"]
        b = record.get("b",None)
        c = record.get("c",None)

        aIdx=a['Idx']
        bIdx=None
        cIdx=None
        if b:
            bIdx=b['Idx']
        if c:
            cIdx=c['Idx']
        _Name = m['_Name']

        if _Name not in mol_buffer:
            # Initialize molecule entry
            mol_buffer[_Name] = dict({
                "molecule": m,
                "atoms": dict({
                    aIdx:a,
                }),
                "bonds": dict()
            })
                    
            if b:
                mol_buffer[_Name]['bonds'][bIdx]=b
            if c:
                mol_buffer[_Name]['atoms'][cIdx]=c
                
        elif mol_buffer[_Name] != True: #if it's true, it means that it's already been yielded
            if aIdx not in mol_buffer[_Name]['atoms']:
                mol_buffer[_Name]['atoms'][aIdx]=a
            if bIdx and bIdx not in mol_buffer[_Name]['bonds']:
                mol_buffer[_Name]['bonds'][bIdx]=b
            if cIdx and cIdx not in mol_buffer[_Name]['atoms']:
                mol_buffer[_Name]['atoms'][cIdx]=c
        else:
            continue
        if len(mol_buffer[_Name]['atoms'])==m['NumAtoms'] and len(mol_buffer[_Name]['bonds'])==m['NumBonds']:
            #print("yielding molecule",_Name,len(mol_buffer[_Name]['atoms']),len(mol_buffer[_Name]['bonds']))
            tmp = (mol_buffer[_Name])
            atoms=tmp['atoms']
            atoms=[atoms[i] for i in atoms]
            bonds=tmp['bonds']
            bonds=[bonds[i] for i in bonds]
            yield atoms, bonds, tmp['molecule']
            mol_buffer[_Name]=True
    for k,v in mol_buffer.items():
        if v != True:
            atoms=v['atoms']
            atoms=[atoms[i] for i in atoms]
            bonds=v['bonds']
            bonds=[bonds[i] for i in bonds]
            #print("yielding molecule",v['_Name'],len(v['atoms']),len(v['bonds']))
            yield atoms, bonds, v['molecule']


def atoms_transform_get_symbol(atoms, x): #should we use X&x or x&x_?
    return atoms, ([(
        *x_, ATOMIC_SYMBOL[atom['Symbol']]
    ) for x_,atom in zip(x,atoms)])

def bonds_transform_get_bondtype(bonds, edge_index, edge_attr):
    return bonds, edge_index, ([(
        *edge_attr_, BONDTYPE[bond['BondType']]
    ) for edge_attr_,bond in zip(edge_attr, bonds)])

def bonds_transform_get_edge_index(bonds, edge_index, edge_attr):
    return bonds, ([(
        *edge_index_, bond['BeginAtomIdx'], bond['EndAtomIdx']
    ) for edge_index_,bond in zip(edge_index, bonds)]), edge_attr

def bonds_transform_add_mirrored_edge_index_and_attr(bonds, edge_index, edge_attr):
    return bonds, tuple(sum([edge_index, [ei[::-1] for ei in edge_index]], [])), tuple(sum([edge_attr, edge_attr], []))

def bonds_transform_transpose_edge_index(bonds, edge_index, edge_attr):
    return bonds, tuple(zip(*edge_index)), edge_attr

def bonds_transform_flatten_edge_attr(bonds, edge_index, edge_attr):
    return bonds, edge_index, tuple(sum((list(ei) for ei in edge_attr),[]))

def atoms_transform_flatten_x(atoms, x):
    return atoms, tuple(sum((list(x_) for x_ in x),[]))

def mol_transform_get_prop(key):
    def inner(mol, y):
        y = (*y, mol[key])
        return mol, y
    return inner

from torch_geometric.data import Data
def mol_to_data(atoms, bonds, mol, atom_transformations, bond_transformations, mol_transformations=None):
    x=tuple(tuple() for _ in range(len(atoms)))
    for t in atom_transformations:
        atoms, x = t(atoms, x)
    
    edge_index=tuple(tuple() for _ in range(len(bonds)))
    edge_attr=tuple(tuple() for _ in range(len(bonds)))
    for t in bond_transformations:
        bonds, edge_index, edge_attr = t(bonds, edge_index, edge_attr)

    if mol_transformations:
        y=()
        for t in mol_transformations:
            mol, y = t(mol, y)
        return Data(x=torch.tensor(x), edge_attr=torch.tensor(edge_attr), 
                    edge_index=torch.tensor(edge_index), y=torch.tensor(y))
    return Data(x=torch.tensor(x), edge_attr=torch.tensor(edge_attr), edge_index=torch.tensor(edge_index))

def mols_to_data(abm_gen, atom_transformations, bond_transformations, mol_transformations=None):
    for atoms, bonds, mol in abm_gen:
        tmp = mol_to_data(atoms, bonds, mol, atom_transformations, bond_transformations, mol_transformations=mol_transformations)
        yield tmp

class EpochedStream:
    def __init__(self, gen, max_epochs):
        self.gen=gen
        self.max_epochs=max_epochs
        self.epoch=0
        self.n=-1
        self.i=0
        self.saves=list()
    def __iter__(self):
        #raise Exception("test")
        return self
    def __next__(self):
        #print('i',self.i)
        if self.epoch==0:
            try:
                self.n+=1
                nxt = next(self.gen)
                self.saves.append(None)
                self.saves[self.n]=nxt
                return nxt
            except:
                print("AAA", len(self.saves))
                #print('saves',self.saves)
                self.epoch+=1
        if self.i>=self.n:
            self.i=0
            self.epoch+=1
        if self.epoch>=self.max_epochs:
            raise StopIteration
        self.i+=1
        return self.saves[self.i-1]

from abc import ABC,abstractmethod
class Task(ABC):
    def __init__(self, atom_transformations, bond_transformations, mol_transformations=None):
        self.atom_transformations=atom_transformations
        self.bond_transformations=bond_transformations
        self.mol_transformations=mol_transformations
    @abstractmethod
    def train(self, model, abm_gen, epochs):
        pass
    @abstractmethod
    def eval(self, model, abm_gen):
        pass

from tqdm.auto import tqdm
class PlogpRegressionTask(Task):
    def __init__(self):
        super().__init__(
            (
                atoms_transform_get_symbol, 
                atoms_transform_flatten_x
            ), 
            (
                bonds_transform_get_bondtype, 
                bonds_transform_get_edge_index, 
                bonds_transform_add_mirrored_edge_index_and_attr, 
                bonds_transform_transpose_edge_index, 
                bonds_transform_flatten_edge_attr
            ),
            (
                mol_transform_get_prop("plogp"),
            )
        )
    def train(self, model: nn.Module, abm_gen, epochs=1):
        optimizer=torch.optim.AdamW(model.parameters())
        MAE=0.0
        n=100
        iterator = tqdm(EpochedStream(
            mols_to_data(
                abm_gen, 
                self.atom_transformations, 
                self.bond_transformations,
                self.mol_transformations
            ),
            max_epochs=epochs
        ))
        for data in iterator:
            output=model(data)
            pred=output.x.mean()
            error=(pred-data.y)**2
            error.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            MAE=(1-1/n)*MAE+error.detach().cpu().item()
            iterator.set_description(f"SSE: {MAE:.2f}")
    def eval(self, model, abm_gen):
        pred=[]
        target=[]
        with torch.no_grad():
            iterator = tqdm(mols_to_data(
                    abm_gen, 
                    self.atom_transformations, 
                    self.bond_transformations,
                    self.mol_transformations
            ))
            for data in iterator:
                output=model(data)
                pred.append(output.x.mean())
                target.append(data.y)
        return pred, target

from neo4j import READ_ACCESS
task=PlogpRegressionTask()
model=RGCN(32,[16,32],1)
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session(default_access_mode=READ_ACCESS) as session:
        with session.begin_transaction() as tx:
            gen1=stream_from_source(tx, "zinc_250k_10k.csv")
            gen2=stream_from_source(tx, "zinc_250k_10k.csv")
            task.train(model, gen1, epochs=1)
            pred, target=task.eval(model, gen2)

plt.scatter(pred,target)
plt.xlabel("pred")
plt.ylabel("target")
plt.title(f"corr={pd.DataFrame({'pred':pred,'target':target}).corr()['pred']['target']:.2f}")