"""Compute unnormalized Zms/Zmol caches with one verified frozen encoder."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from .config import load, path
from .inputs import metadata, table, graph, graph_batch, spectrum_batch
from .model import build, load_weights
from utils.artifacts import resolve, sha256, publish
from utils.config import external_root
from utils.formula import SPECTRUM_PRECURSOR_INDEX
from utils.collision_energy import ENERGY_ENCODING


def supervised_rows(filename):
    rows = table(filename)
    if not rows or any(r.get('split') not in ('train', 'val') for r in rows):
        raise ValueError(f'{filename}: graph rows must contain only train/val and include a split column')
    return rows


def write_cache(directory, tower, keys, batches, checkpoint, source):
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f'{tower}: empty or duplicate cache keys')
    directory.mkdir()
    array = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype=np.float32, shape=(len(keys),512))
    count = 0
    for z in batches:
        z = z.detach().float().cpu()
        if z.ndim != 2 or z.shape[1] != 512 or not torch.isfinite(z).all() or (z.norm(dim=1) == 0).any():
            raise ValueError(f'{tower}: invalid encoder output')
        array[count:count+len(z)] = z.numpy(); count += len(z)
    if count != len(keys): raise ValueError('Incomplete embedding export')
    array.flush(); del array
    (directory / 'keys.json').write_text(json.dumps(keys), encoding='utf-8')
    manifest = dict(tower=tower, dim=512, dtype='float32', normalization='NONE', count=count,
                    encoder_sha256=sha256(checkpoint), encoder_checkpoint=str(checkpoint), source=source,
                    sha256={name: sha256(directory/name) for name in ('features.npy', 'keys.json')})
    (directory / 'CACHE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def export_caches(cfg, checkpoint, output, graph_rows, batch_size=32, device='cpu'):
    if batch_size < 1: raise ValueError('batch_size must be positive')
    selected = cfg['datasets'][cfg['dataset']]
    root = path(selected['data_root'])
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(f'Embedding output exists: {output}; use a new directory')
    checkpoint = resolve(checkpoint, 'encoder')
    digest = sha256(checkpoint)
    model = build().to(device); load_weights(model, checkpoint)
    model.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = metadata(root)
    excluded = [dict(spec=r['spec'], reason='UNSUPPORTED_ADDUCT') for r in rows if r['ionization'] not in SPECTRUM_PRECURSOR_INDEX]
    rows = [r for r in rows if r['ionization'] in SPECTRUM_PRECURSOR_INDEX]
    # Both the pretraining pool and MS teacher targets must be encoded in this same space.
    ms_rows = path(selected['prepared_dir']) / 'manifest.tsv'
    sources = [Path(graph_rows).resolve(), ms_rows]
    keys = sorted({graph(r['smiles'])[0] for source in sources for r in supervised_rows(source)})
    source = dict(dataset=cfg['dataset'], labels_sha256=sha256(root/'labels.tsv'),
                  split_sha256=sha256(root/'split.tsv'), graph_rows={str(p):sha256(p) for p in sources},
                  subformula_sha256={r['spec']:sha256(root/'subformulae/default_subformulae'/(r['spec']+'.json')) for r in rows},
                   spectrum_truth_used=False, zms_excluded=excluded,
                   energy_encoding=ENERGY_ENCODING,
                   energy_bins={r['spec']: r['energy_bin'] for r in rows})
    def spectra():
        with torch.no_grad():
            for i in range(0,len(rows),batch_size):
                sos, formula, mask = spectrum_batch(rows[i:i+batch_size],root,device)
                yield model.ms_encoder(sos,formula,mask)
    def molecules():
        with torch.no_grad():
            for i in range(0,len(keys),batch_size):
                yield model.graph_encoder(*graph_batch(keys[i:i+batch_size],device))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp)/'embeddings';staged.mkdir()
        write_cache(staged/'zms','zms',[r['spec'] for r in rows],spectra(),checkpoint,source)
        write_cache(staged/'zmol','zmol',keys,molecules(),checkpoint,source)
        if sha256(checkpoint) != digest: raise ValueError('Encoder changed during export')
        publish(staged/'encoder.json',checkpoint,'encoder')
        staged.rename(output)
    print(f'Exported {len(rows)} Zms and {len(keys)} Zmol vectors using encoder {digest}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/encoder.yml')
    parser.add_argument('--dataset',choices=['msg','nplib1'])
    parser.add_argument('--checkpoint',help='Encoder weight file or pointer; default: completed fine-tuning output')
    parser.add_argument('--graph-rows',type=Path,help='CSV/TSV with smiles and train/val split for Graph2Mol')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args();cfg=load(args.config,dataset=args.dataset)
    selected=cfg['datasets'][cfg['dataset']]
    checkpoint=args.checkpoint or path(selected['weights_dir'])/'latest.json'
    output=args.output or external_root()/cfg['dataset']/'embeddings'
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(cfg['runtime']['threads'])
    export_caches(cfg,checkpoint,output,args.graph_rows or path(selected['graph_rows']),args.batch_size,device)


if __name__=='__main__':main()
