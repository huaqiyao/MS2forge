"""Training-only true graph access; no generator imports this module."""
import csv,json
from dataclasses import replace
from pathlib import Path
import torch
from torch.utils.data import Dataset,Sampler
from .graph import graph_from_smiles
from .formula import parse_formula,acquisition_indices,InputFailure,MODEL_ATOMIC_NUMBERS
from .permutation import permute_supervision
from .randomness import generator

def hill_supervision(graph,formula,*,seed=2026,epoch=0,occurrence=0):
    """ALL heavy atoms shuffled on every Train visit, labels mapped together."""
    spec=parse_formula(formula,adapter='graph2mol')
    if sorted(graph.node_type.tolist())!=sorted(spec.node_types):raise InputFailure('FORMULA_GRAPH_MULTISET',graph.sample_id)
    perm=torch.randperm(len(graph.node_type),generator=generator(graph.sample_id,seed,epoch,occurrence,'all_heavy_train'))
    mapped=permute_supervision(graph.node_type,graph.halfedge_index,graph.halfedge_type,perm)
    return replace(graph,node_type=mapped['node_type'],halfedge_index=mapped['halfedge_index'],halfedge_type=mapped['halfedge_type'],mapping=mapped['mapping'],visit_id=int(occurrence))

class HillTrainingDataset(Dataset):
    def __init__(self,rows,*,stage,zms,zmol,seed=2026,engineering=False):
        if stage not in ('graph2mol','ms2mol'):raise ValueError(stage)
        if engineering and len(rows)>8:raise ValueError('Engineering adapter budget: <=8 Train/Val records')
        if getattr(zmol,'role',None)!='zmol':raise ValueError('Training requires declared Zmol by sample_id')
        if stage=='ms2mol' and getattr(zms,'role',None)!='zms':raise ValueError('MS training requires Zms')
        if any(r.get('split') not in ('train','val') for r in rows):raise ValueError('Train/Val only, never Test')
        if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate supervised IDs')
        self.rows=rows;self.stage=stage;self.zms=zms;self.zmol=zmol;self.seed=seed
    @classmethod
    def from_file(cls,path,**kwargs):
        p=Path(path)
        if p.suffix=='.jsonl':rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        else:
            with p.open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        return cls(rows,**kwargs)
    def __len__(self):return len(self.rows)
    def __getitem__(self,token):
        index,epoch,occurrence=token if isinstance(token,tuple) else (token,0,token)
        row=self.rows[index];sid=str(row['sample_id'])
        adduct=row.get('adduct')
        if self.stage=='graph2mol' and not adduct:adduct='[M+H]+'
        inst,ion=acquisition_indices(row.get('instrument','NONE'),adduct,row.get('adapter','msg'))
        zmol=self.zmol.get_supervised(sid,row['smiles'])
        zms=self.zms.get(sid) if self.stage=='ms2mol' else torch.zeros_like(zmol)
        for z in (zmol,zms):
            if z.shape!=(512,) or not torch.isfinite(z).all():raise ValueError('Invalid training feature')
        graph=graph_from_smiles(sid,row['smiles'],zms,zmol,split=row['split'],instrument_idx=inst,ionization_idx=ion)
        # Formula is explicitly supplied by supervised producer, not inferred
        # through a generation fallback. Check FULL H formula during TRAIN.
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        molecule=Chem.MolFromSmiles(row['smiles'])
        if molecule is None:raise InputFailure('TRAIN_SMILES_PARSE_FAILED',sid)
        actual=rdMolDescriptors.CalcMolFormula(molecule)
        if parse_formula(actual,adapter='graph2mol').counts!=parse_formula(row['formula'],adapter='graph2mol').counts:
            raise InputFailure('TRAIN_FORMULA_MISMATCH',sid)
        return hill_supervision(graph,row['formula'],seed=self.seed,epoch=epoch,occurrence=occurrence)

class VisitSampler(Sampler):
    """Committed consumption cursor, deterministic across workers and resume.

    Rank/world are part of resume identity (not noise keys). DDP padding is
    training-only; evaluation uses no-padding shards. Global visit token follows
    the sample's position in the shared epoch order.
    """
    def __init__(self,n,seed=2026,rank=0,world=1):
        if n<1 or not 0<=rank<world:raise ValueError('Bad sampler size/rank')
        self.n=n;self.seed=seed;self.rank=rank;self.world=world;self.epoch=0;self.position=0
    def _indices(self):
        order=torch.randperm(self.n,generator=generator(self.seed,self.epoch,'train_epoch')).tolist()
        total=((self.n+self.world-1)//self.world)*self.world
        order=(order*((total+self.n-1)//self.n))[:total]
        return [(order[i],self.epoch,self.epoch*total+i) for i in range(self.rank,total,self.world)]
    def __iter__(self):return iter(self._indices()[self.position:])
    def __len__(self):return len(self._indices())-self.position
    def commit(self,count):
        self.position+=count
        if self.position>len(self._indices()):raise ValueError('Commit exceeds epoch')
    def next_epoch(self):
        if self.position!=len(self._indices()):raise ValueError('Cannot skip unconsumed visits')
        self.epoch+=1;self.position=0
    def state_dict(self):return vars(self).copy()
    def load_state_dict(self,state):
        if any(state[k]!=getattr(self,k) for k in ('n','seed','rank','world')):raise ValueError('Resume sampler identity mismatch')
        self.epoch=int(state['epoch']);self.position=int(state['position'])
        if not 0<=self.position<=len(self._indices()):raise ValueError('Invalid resume cursor')
