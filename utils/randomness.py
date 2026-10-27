"""Stateless sample/draw/step/stream noise, explicitly mapped to physical slots."""
import hashlib
import json
import torch
RNG_SCHEME = "sample_draw_step_stream_sha256_v2"

def seed_from(*parts):
    payload = json.dumps([RNG_SCHEME,*parts],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little') & ((1<<63)-1)

def generator(*parts):
    return torch.Generator(device='cpu').manual_seed(seed_from(*parts))

def order_permutation(n, sample_id, order_seed, epoch=0, occurrence=0):
    return torch.randperm(n,generator=generator(sample_id,order_seed,epoch,occurrence,'node_order'))

class ManagedNoise:
    def __init__(self, batch, sampling_seed):
        self.batch=batch; self.seed=int(sampling_seed)

    def _normal(self, entity, step, shape, device, dtype, stream):
        pieces=[]
        maps = self.batch.node_to_base if entity=='node' else self.batch.edge_to_base
        expected=0
        for item,mapping,draw in zip(self.batch.items,maps,self.batch.draw_ids):
            count=len(mapping); expected+=count
            g=generator(item.sample_id,self.seed,int(draw),int(step),stream)
            base=torch.randn((count,*shape[1:]),generator=g,dtype=torch.float64)
            pieces.append(base[mapping.cpu()])
        if expected!=shape[0]:raise ValueError('Noise shape and slot mapping disagree')
        result=torch.cat(pieces,dim=0) if pieces else torch.empty(shape,dtype=torch.float64)
        return result.to(device=device,dtype=dtype)

    def edge(self, step, shape, device, dtype, stream='edge_gaussian'):
        return self._normal('edge',step,shape,device,dtype,stream)

    def node(self, step, shape, device, dtype, stream='node_gaussian'):
        return self._normal('node',step,shape,device,dtype,stream)

    def time(self, step, device, dtype):
        values=[]
        for item,draw in zip(self.batch.items,self.batch.draw_ids):
            values.append(torch.rand((),generator=generator(item.sample_id,self.seed,draw,step,'train_time'),dtype=torch.float64))
        return torch.stack(values).to(device=device,dtype=dtype)
