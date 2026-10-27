"""Generation, training supervision and scoring are separate types/interfaces."""
from dataclasses import dataclass,replace
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .formula import parse_formula,acquisition_indices,InputFailure,MODEL_ATOMIC_NUMBERS
from .permutation import relabel,permute_supervision
from .randomness import order_permutation

ALLOWED_INPUT_FIELDS=('sample_id','formula','formula_kind','formula_source','instrument','adduct','split',
                      'spectrum_path','subformula_path','precursor_mz')

@dataclass(frozen=True)
class GenerationInput:
    sample_id: str
    formula: object
    formula_source: str
    instrument_idx: int
    ionization_idx: int
    zms: torch.Tensor
    node_type: torch.Tensor
    halfedge_index: torch.Tensor
    node_to_base: torch.Tensor
    edge_to_base: torch.Tensor
    spectrum_path: str=''
    split: str=''
    draw_id: int=0
    order_basis: str='neutral_formula_slots'

@dataclass(frozen=True)
class FailedInput:
    sample_id: str
    reason: str

class StrictFormulaDataset(Dataset):
    def __init__(self,rows,cache,*,adapter='msg',max_atoms=150,order_seed=None):
        if adapter not in ('msg','nplib1'):raise InputFailure('GENERATION_ADAPTER','Only audited spectrum adapters')
        if getattr(cache,'role',None)!='zms':raise InputFailure('GENERATION_CACHE_ROLE','Generation requires declared Zms, never Zmol')
        self.rows=rows; self.cache=cache; self.adapter=adapter; self.max_atoms=max_atoms; self.order_seed=order_seed
    @classmethod
    def from_file(cls,path,cache,**kwargs):
        # Dedicated generation file, never a reference/labels file.
        path=Path(path)
        if path.suffix=='.jsonl':rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        else:
            with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        unexpected=set().union(*(set(r)-set(ALLOWED_INPUT_FIELDS) for r in rows))
        if unexpected:raise InputFailure('GENERATION_SCHEMA',','.join(sorted(unexpected)))
        return cls(rows,cache,**kwargs)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        # No key iteration or access to forbidden fields: access-sentinel tests
        # exercise the same routine with a hostile mapping.
        sid=str(row['sample_id'])
        try:
            formula=parse_formula(row.get('formula'),adapter=self.adapter,formula_kind=row.get('formula_kind','neutral'),max_atoms=self.max_atoms)
            source=row.get('formula_source','')
            if not source:raise InputFailure('FORMULA_SOURCE_MISSING',sid)
            # A missing acquisition annotation is a failed query, never H+.
            adduct=row.get('adduct')
            inst,ion=acquisition_indices(row.get('instrument','NONE'),adduct,self.adapter)
            zms=self.cache.get(sid)
            if zms is None or zms.numel()!=512 or not torch.isfinite(zms).all():raise InputFailure('INVALID_ZMS',sid)
            node=torch.tensor(formula.node_types,dtype=torch.long)
            if len(node)<2:raise InputFailure('NO_HALFEDGE_FROZEN_SAMPLER',sid)
            perm=torch.arange(len(node)) if self.order_seed is None else order_permutation(len(node),sid,self.order_seed)
            r=relabel(perm)
            return GenerationInput(sid,formula,str(source),inst,ion,zms.reshape(512),node[perm],r.edges,
                                   r.new_to_old_node,r.new_to_old_edge,row.get('spectrum_path',''),row.get('split',''))
        except (InputFailure,KeyError) as error:
            return FailedInput(sid,getattr(error,'code',str(error)))

@dataclass
class GenerationBatch:
    items: list
    node_type: torch.Tensor
    halfedge_index: torch.Tensor
    node_type_batch: torch.Tensor
    halfedge_type_batch: torch.Tensor
    cond_emb_cached: torch.Tensor
    instrument_type_idx_batch: torch.Tensor
    ionization_type_idx_batch: torch.Tensor
    node_to_base: list
    edge_to_base: list
    draw_ids: list
    node_offsets: list
    edge_offsets: list
    def to(self,device,dtype=torch.float32):
        for name in ('node_type','halfedge_index','node_type_batch','halfedge_type_batch',
                     'instrument_type_idx_batch','ionization_type_idx_batch'):
            setattr(self,name,getattr(self,name).to(device))
        self.cond_emb_cached=self.cond_emb_cached.to(device=device,dtype=dtype)
        return self

def collate_generation(items):
    if not items or any(not isinstance(x,GenerationInput) for x in items):raise TypeError('Only successful label-free inputs can be collated')
    nodes=[]; edges=[]; bn=[]; be=[]; no=[0]; eo=[0]
    for i,x in enumerate(items):
        n=len(x.node_type); e=x.halfedge_index.shape[1]
        nodes.append(x.node_type); edges.append(x.halfedge_index+no[-1]); bn.append(torch.full((n,),i,dtype=torch.long)); be.append(torch.full((e,),i,dtype=torch.long))
        no.append(no[-1]+n); eo.append(eo[-1]+e)
    return GenerationBatch(items,torch.cat(nodes),torch.cat(edges,dim=1),torch.cat(bn),torch.cat(be),
                           torch.stack([x.zms for x in items]),torch.tensor([x.instrument_idx for x in items]),
                           torch.tensor([x.ionization_idx for x in items]),[x.node_to_base for x in items],
                           [x.edge_to_base for x in items],[x.draw_id for x in items],no,eo)

@dataclass(frozen=True)
class SupervisedGraph:
    sample_id: str
    node_type: torch.Tensor
    halfedge_index: torch.Tensor
    halfedge_type: torch.Tensor
    zms: torch.Tensor
    zmol: torch.Tensor
    instrument_idx: int=2
    ionization_idx: int=0
    mapping: object=None
    visit_id: int=0

def graph_from_smiles(sample_id,smiles,zms,zmol,*,split='train',instrument_idx=2,ionization_idx=0):
    if split not in ('train','val'):raise InputFailure('SUPERVISION_SPLIT','Test is forbidden in training/validation supervision')
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise InputFailure('INVALID_TRAIN_STRUCTURE',sample_id)
    try:canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
    except Exception as e:raise InputFailure('TRAIN_CANONICAL_FAILED',sample_id) from e
    mol=Chem.MolFromSmiles(canonical)
    if mol is None:raise InputFailure('TRAIN_CANONICAL_REPARSE_FAILED',sample_id)
    atoms=[a.GetAtomicNum() for a in mol.GetAtoms()]
    if any(z not in MODEL_ATOMIC_NUMBERS for z in atoms):raise InputFailure('UNSUPPORTED_TRAIN_ELEMENT',sample_id)
    node=torch.tensor([MODEL_ATOMIC_NUMBERS.index(z) for z in atoms]); edge=torch.triu_indices(len(node),len(node),1)
    dense=torch.zeros(len(node),len(node),dtype=torch.long)
    kinds={Chem.BondType.SINGLE:1,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3,Chem.BondType.AROMATIC:4}
    for b in mol.GetBonds():dense[b.GetBeginAtomIdx(),b.GetEndAtomIdx()]=dense[b.GetEndAtomIdx(),b.GetBeginAtomIdx()]=kinds[b.GetBondType()]
    return SupervisedGraph(sample_id,node,edge,dense[edge[0],edge[1]],zms,zmol,instrument_idx,ionization_idx,relabel(torch.arange(len(node))))

def randomly_relabel_graph(graph,seed,epoch=0,occurrence=0):
    perm=order_permutation(len(graph.node_type),graph.sample_id,seed,epoch,occurrence)
    result=permute_supervision(graph.node_type,graph.halfedge_index,graph.halfedge_type,perm)
    return replace(graph,node_type=result['node_type'],halfedge_index=result['halfedge_index'],
                   halfedge_type=result['halfedge_type'],mapping=result['mapping'])

def collate_supervised(graphs,stage):
    # Only this object carries truth edge labels; never accepted by generation.
    items=[]
    for x in graphs:
        r=x.mapping or relabel(torch.arange(len(x.node_type)))
        items.append(GenerationInput(x.sample_id,None,'training supervision',x.instrument_idx,x.ionization_idx,
                                     x.zms if stage=='ms2mol' else x.zmol,x.node_type,x.halfedge_index,
                                     r.new_to_old_node,r.new_to_old_edge,draw_id=x.visit_id))
    batch=collate_generation(items)
    batch.halfedge_type=torch.cat([x.halfedge_type for x in graphs]); batch.zmol_target=torch.stack([x.zmol for x in graphs])
    return batch
