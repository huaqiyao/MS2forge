"""Read-only qualified full pool. Random relabel follows each logical visit."""
import json,sqlite3
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset
from utils.graph import SupervisedGraph
from utils.permutation import relabel
from utils.supervision import hill_supervision
class PackedGraphDataset(Dataset):
    def __init__(self,root,seed=2026):
        self.root=Path(root);self.manifest=json.loads((self.root/'CACHE_MANIFEST.json').read_text());self.seed=seed
        self.features=np.load(self.root/'features.npy',mmap_mode='r');self.counts=np.load(self.root/'node_counts.npy',mmap_mode='r');self.con=None
        if self.features.shape!=(self.manifest['qualified_rows'],512):raise ValueError('Full feature coverage mismatch')
    def connection(self):
        if self.con is None:self.con=sqlite3.connect('file:'+str(self.root/'records.sqlite')+'?mode=ro',uri=True,check_same_thread=False)
        return self.con
    def __len__(self):return len(self.counts)
    def __getitem__(self,token):
        idx,epoch,visit=token if isinstance(token,tuple) else (int(token),0,int(token));row=self.connection().execute('SELECT sample_id,formula,n,nodes,bonds FROM records WHERE idx=?',(idx,)).fetchone()
        if row is None:raise KeyError('Missing qualified training row '+str(idx))
        sid,formula,n,nodes,bonds=row;node=torch.from_numpy(np.frombuffer(nodes,dtype=np.uint8).astype(np.int64));sparse=np.frombuffer(bonds,dtype=np.uint8).reshape(-1,3);edge=torch.triu_indices(n,n,1);dense=torch.zeros(n,n,dtype=torch.long)
        if len(sparse):
            b=torch.tensor(sparse.astype(np.int64));dense[b[:,0],b[:,1]]=b[:,2];dense[b[:,1],b[:,0]]=b[:,2]
        z=torch.tensor(np.array(self.features[idx]),dtype=torch.float32)
        if not torch.isfinite(z).all() or z.norm()==0:raise ValueError('Nonfinite/missing Zmol; never skip')
        graph=SupervisedGraph(sid,node,edge,dense[edge[0],edge[1]],torch.zeros_like(z),z,mapping=relabel(torch.arange(n)))
        return hill_supervision(graph,formula,seed=self.seed,epoch=epoch,occurrence=visit)
class GlobalVisitSchedule:
    """Logical global batch shared by all ranks; only last-epoch padding repeats."""
    def __init__(self,n,batch,seed,rank,world):
        if batch%world:raise ValueError('Global batch must divide world')
        self.n=n;self.batch=batch;self.seed=seed;self.rank=rank;self.world=world;self.epoch=0;self.position=0;self.order=None
    def indices(self):
        if self.order is None:
            g=torch.Generator().manual_seed(self.seed+self.epoch*1000003);self.order=torch.randperm(self.n,generator=g).tolist()
        start=self.position;global_ids=[self.order[(start+j)%self.n] for j in range(self.batch)];total=((self.n+self.batch-1)//self.batch)*self.batch
        return [(global_ids[j],self.epoch,self.epoch*total+start+j) for j in range(self.rank,self.batch,self.world)]
    def commit(self):
        self.position+=self.batch
        if self.position>=self.n:self.epoch+=1;self.position=0;self.order=None
    def state_dict(self):return {k:getattr(self,k) for k in ['n','batch','seed','rank','world','epoch','position']}
    def load_state_dict(self,state):
        for k in ['n','batch','seed','rank','world']:
            if state[k]!=getattr(self,k):raise ValueError('Resume logical visit/world mismatch '+k)
        self.epoch=state['epoch'];self.position=state['position'];self.order=None


"""Same full logical visits; target graph and Zmol live only in supervised class."""
import json,sqlite3
from pathlib import Path
import numpy as np,torch
from utils.graph import SupervisedGraph
from utils.supervision import hill_supervision
from utils.permutation import relabel
class PackedMSDataset:
 def __init__(self,root,seed=2026,zms_root=None,zmol_root=None):
  self.root=Path(root);self.manifest=json.loads((self.root/'CACHE_MANIFEST.json').read_text());self.seed=seed;self.con=None;self.counts=np.load(self.root/'node_counts.npy',mmap_mode='r');self.zms=np.load(Path(zms_root or self.manifest['zms_root'])/'features.npy',mmap_mode='r');self.zmol=np.load(Path(zmol_root or self.manifest['zmol_root'])/'features.npy',mmap_mode='r')
 def __len__(self):return len(self.counts)
 def connection(self):
  if self.con is None:self.con=sqlite3.connect('file:'+str(self.root/'records.sqlite')+'?mode=ro',uri=True,check_same_thread=False)
  return self.con
 def __getitem__(self,token):
  idx,epoch,visit=token if isinstance(token,tuple) else (int(token),0,int(token));r=self.connection().execute('SELECT sample_id,formula,n,nodes,bonds,zms_index,zmol_index,instrument,adduct FROM records WHERE idx=?',(idx,)).fetchone()
  if r is None:raise KeyError('Missing supervised MS query; never skip')
  sid,formula,n,nodebytes,bondbytes,si,gi,inst,ion=r;nodes=torch.tensor(np.frombuffer(nodebytes,dtype=np.uint8).astype(np.int64));sparse=np.frombuffer(bondbytes,dtype=np.uint8).reshape(-1,3);edge=torch.triu_indices(n,n,1);dense=torch.zeros(n,n,dtype=torch.long)
  if len(sparse):
   b=torch.tensor(sparse.astype(np.int64));dense[b[:,0],b[:,1]]=b[:,2];dense[b[:,1],b[:,0]]=b[:,2]
  zms=torch.tensor(np.array(self.zms[si]));zmol=torch.tensor(np.array(self.zmol[gi]));assert zms.shape==zmol.shape==(512,) and torch.isfinite(zms).all() and torch.isfinite(zmol).all() and zms.norm()>0 and zmol.norm()>0
  graph=SupervisedGraph(sid,nodes,edge,dense[edge[0],edge[1]],zms,zmol,inst,ion,mapping=relabel(torch.arange(n)));return hill_supervision(graph,formula,seed=self.seed,epoch=epoch,occurrence=visit)
class ZmsMmapCache:
 role='zms'
 def __init__(self,root):
  root=Path(root);m=json.loads((root/'CACHE_MANIFEST.json').read_text());assert m['tower']=='zms' and m['dtype']=='float32' and m['normalization']=='NONE';self.keys={s:i for i,s in enumerate(json.loads((root/'keys.json').read_text()))};self.values=np.load(root/'features.npy',mmap_mode='r');self.queries=[]
 def get(self,sid):
  self.queries.append(sid)
  if sid not in self.keys:raise KeyError('MISSING_CLEAN39K_ZMS')
  return torch.tensor(np.array(self.values[self.keys[sid]]),dtype=torch.float32)
