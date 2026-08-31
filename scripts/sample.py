"""MS2Forge module."""
import os
import sys
import argparse
import time
import json
import random
import hashlib
from collections import Counter

sys.path.append('.')

import torch
import torch.distributed as dist
import yaml
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.dataset import DiffMSMSGDataset, ensure_cond_emb_cache
from utils.transforms import make_msg_diffms_collate_with_cache
from utils.eval_utils import (
    build_candidate_cache_record,
    topk_hit_for_mol as _topk_hit_for_mol_dispatcher,
)
from models.model import FLASH


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def parse_args():
    p = argparse.ArgumentParser('FLASH ms2mol inference and evaluation')
    p.add_argument('--config',        type=str, default='./configs/sample.yml')
    p.add_argument('--ms2mol_ckpt',   type=str, default=None,
                   help='MS2Mol or joint checkpoint path; defaults to cfg.ckpt.ms2mol')
    p.add_argument('--align_ckpt',    type=str, default=None,
                   help='Alignment checkpoint used to build the Zms cache')
    p.add_argument('--adapter_ckpt', type=str, default=None,
                   help='Optional checkpoint that overrides only zms_adapter.* parameters')
    p.add_argument('--indices_json', type=str, default=None,
                   help='Fixed validation-spectrum manifest; disables subset_ratio sampling')
    p.add_argument('--device',        type=str, default='auto')
    p.add_argument('--split',         type=str, default=None)
    p.add_argument('--batch_size',    type=int, default=None)
    p.add_argument('--n_samples',     type=int, default=None)
    p.add_argument('--n_timesteps',   type=int, default=None)
    p.add_argument('--subset_ratio',  type=float, default=None)
    p.add_argument('--seed',          type=int, default=None)
    p.add_argument('--topk',          type=str, default=None)
    p.add_argument('--output_dir',    type=str, default=None)
    p.add_argument('--save_per_spec', action='store_true')
    p.add_argument('--no_save_per_spec', action='store_true',
                   help='Disable per-spectrum outputs even when save_per_spec=true')
    p.add_argument('--save_candidate_cache', action='store_true',
                   help='Save per-spectrum candidate frequencies for offline reevaluation')
    p.add_argument('--no_save_candidate_cache', action='store_true',
                   help='Disable the candidate cache even when save_candidate_cache=true')
    p.add_argument('--candidate_cache_jsonl', type=str, default=None,
                   help='Custom candidate-cache JSONL path; use one file per DDP rank')
    p.add_argument('--resume_progress', action='store_true',
                   help='Resume from compact JSONL progress and restore cumulative metrics')
    p.add_argument('--progress_jsonl', type=str, default=None,
                   help='Custom compact-progress JSONL path; use one file per DDP rank')
    p.add_argument('--inner_chunk',   type=int, default=None)
    p.add_argument('--condition_mode', type=str, default=None,
                   choices=['real', 'default'],
                   help='BFN labels: real instrument/ionization or default NONE + [M+H]+')
    p.add_argument('--condition_source', type=str, default=None,
                   choices=['zms', 'zmol'],
                   help='Condition-vector source; Zmol selects oracle evaluation and disables the adapter')
    p.add_argument('--eval_mode',     type=str, default=None,
                   choices=['strict', 'isomorphic', 'inchikey', 'diffms_inchi', 'diffms_2d', None],
                   help='Evaluation method; stage II uses diffms_2d')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0,
                   help='torchrun compatibility argument, normally supplied through LOCAL_RANK')
    return p.parse_args()


def topk_hit_for_mol(pred_seqs, true_seq, topk_list, mode='strict',
                      node_types=None, halfedge_index=None, atomic_numbers=None,
                      true_smiles=None):
    return _topk_hit_for_mol_dispatcher(
        pred_seqs, true_seq, topk_list,
        mode=mode,
        node_types=node_types,
        halfedge_index=halfedge_index,
        atomic_numbers=atomic_numbers,
        true_smiles=true_smiles,
    )


def non_adapter_state_sha256(model):
    """Bitwise digest of the decoder/shared path, excluding zms_adapter.*."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith('zms_adapter.'):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_compact_progress(path, topk_list):
    """Load completed batch records from a compact progress jsonl.

    A batch only counts after a trailing {"type": "batch_done"} marker, so
    crashes in the middle of a batch do not create duplicate metric counts.
    """
    done_batches = set()
    mol_records = {}
    if not path or not os.path.exists(path):
        return {
            'done_batches': done_batches,
            'topk_correct': {k: 0 for k in topk_list},
            'total_mols': 0,
            'edge_correct': 0,
            'total_edges': 0,
            'bond_correct': 0,
            'bond_total': 0,
        }

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch_idx = rec.get('batch_idx')
            if batch_idx is None:
                continue
            batch_idx = int(batch_idx)
            if rec.get('type') == 'batch_done':
                done_batches.add(batch_idx)
            else:
                mol_records.setdefault(batch_idx, []).append(rec)

    topk_correct = {k: 0 for k in topk_list}
    total_mols = 0
    total_edges = 0
    edge_correct = 0
    bond_total = 0
    bond_correct = 0
    seen = set()
    for batch_idx in sorted(done_batches):
        for rec in mol_records.get(batch_idx, []):
            mol_idx = int(rec.get('mol_local_idx', -1))
            key = (batch_idx, mol_idx)
            if mol_idx < 0 or key in seen:
                continue
            seen.add(key)
            hits = rec.get('hits', {})
            for k in topk_list:
                if bool(hits.get(str(k), hits.get(k, False))):
                    topk_correct[k] += 1
            total_mols += 1
            edge_correct += int(rec.get('edge_correct', 0))
            total_edges += int(rec.get('total_edges', 0))
            bond_correct += int(rec.get('bond_correct', 0))
            bond_total += int(rec.get('bond_total', 0))

    return {
        'done_batches': done_batches,
        'topk_correct': topk_correct,
        'total_mols': total_mols,
        'edge_correct': edge_correct,
        'total_edges': total_edges,
        'bond_correct': bond_correct,
        'bond_total': bond_total,
    }


def main():
    args = parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.config)))
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = 0
    world_size = 1
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    is_main = (rank == 0)

    if args.split is not None:        cfg.evaluate.split = args.split
    if args.batch_size is not None:   cfg.evaluate.batch_size = args.batch_size
    if args.n_samples is not None:    cfg.evaluate.n_samples = args.n_samples
    if args.n_timesteps is not None:  cfg.evaluate.n_timesteps = args.n_timesteps
    if args.subset_ratio is not None: cfg.evaluate.subset_ratio = args.subset_ratio
    if args.seed is not None:         cfg.evaluate.seed = args.seed
    if args.output_dir is not None:   cfg.evaluate.output_dir = args.output_dir
    if args.topk is not None:
        cfg.evaluate.topk = [int(x) for x in args.topk.split(',')]
    if args.inner_chunk is not None:
        cfg.evaluate.inner_chunk = args.inner_chunk
    if args.eval_mode is not None:
        cfg.evaluate.eval_mode = args.eval_mode
    if args.condition_mode is not None:
        cfg.evaluate.condition_mode = args.condition_mode
    if args.condition_source is not None:
        cfg.evaluate.condition_source = args.condition_source
    cfg.evaluate.inner_chunk = int(cfg.evaluate.get('inner_chunk', cfg.evaluate.n_samples))
    condition_mode = cfg.evaluate.get('condition_mode', 'real')
    condition_source = cfg.evaluate.get('condition_source', 'zms')
    if condition_mode not in ('real', 'default'):
        raise ValueError(f'condition_mode must be real or default; received {condition_mode!r}')
    if condition_source not in ('zms', 'zmol'):
        raise ValueError(f'condition_source must be zms or zmol; received {condition_source!r}')

    ms2mol_ckpt = args.ms2mol_ckpt or cfg.ckpt.ms2mol
    align_ckpt = args.align_ckpt or cfg.ckpt.get('align', './checkpoints/align/align.pt')
    assert os.path.exists(ms2mol_ckpt), f'MS2Mol checkpoint does not exist: {ms2mol_ckpt}'
    assert os.path.exists(align_ckpt), f'Alignment checkpoint does not exist: {align_ckpt}'
    if args.adapter_ckpt is not None:
        assert os.path.exists(args.adapter_ckpt), f'Adapter checkpoint does not exist: {args.adapter_ckpt}'
    if args.indices_json is not None:
        assert os.path.exists(args.indices_json), f'Index manifest does not exist: {args.indices_json}'

    cfg.model.stage = 'ms2mol'

    seed = int(cfg.evaluate.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if distributed:
        rank_seed = seed + rank * 1000003
        random.seed(rank_seed); np.random.seed(rank_seed); torch.manual_seed(rank_seed)

    if distributed and torch.cuda.is_available():
        device = f'cuda:{local_rank}'
    elif args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print('[WARNING] CUDA is unavailable; switching to CPU')
            device = 'cpu'

    os.makedirs(cfg.evaluate.output_dir, exist_ok=True)

    if is_main:
        print('=' * 70)
        print('FLASH ms2mol inference and evaluation')
        print('=' * 70)
        print(f'  ms2mol_ckpt    : {ms2mol_ckpt}')
        print(f'  adapter_ckpt   : {args.adapter_ckpt or "<none>"}')
        print(f'  indices_json   : {args.indices_json or "<random subset>"}')
        print(f'  align_ckpt     : {align_ckpt}')
        print(f'  device         : {device}')
        print(f'  distributed    : {distributed} (world_size={world_size})')
        print(f'  split          : {cfg.evaluate.split}')
        print(f'  batch_size     : {cfg.evaluate.batch_size}')
        print(f'  n_samples/spectrum   : {cfg.evaluate.n_samples}')
        print(f'  inner_chunk    : {cfg.evaluate.inner_chunk}')
        print(f'  n_timesteps    : {cfg.evaluate.n_timesteps}')
        print(f'  subset_ratio   : {cfg.evaluate.subset_ratio}')
        print(f'  topk           : {cfg.evaluate.topk}')
        print(f'  eval_mode      : {cfg.evaluate.get("eval_mode", "isomorphic")}')
        print(f'  condition      : source={condition_source}, labels={condition_mode}')
        print(f'  resume_progress: {args.resume_progress}')
        print(f'  candidate_cache: {not args.no_save_candidate_cache and (args.save_candidate_cache or cfg.evaluate.get("save_candidate_cache", False))}')
        print(f'  seed           : {seed}')
        print('=' * 70)

    # [1/3] zms cache
    if is_main:
        print('\n[1/3] Building or loading the Zms cache...')
    zms_cache = ensure_cond_emb_cache(
        stage='ms2mol',
        align_ckpt_path=align_ckpt,
        msg_root=cfg.dataset.root,
        cache_dir=getattr(cfg.dataset, 'cache_dir', './data/cache'),
        device=device,
        batch_size=128,
    )
    if is_main:
        print(f'  Zms cache: {len(zms_cache)} entries')

    # [2/3] dataset
    if is_main:
        print('\n[2/3] Loading MSG ...')
    ds = DiffMSMSGDataset(root=cfg.dataset.root)
    split_name = cfg.evaluate.split
    full_subset = ds.subsets[split_name]
    full_indices = list(full_subset.indices)
    total_n = len(full_indices)
    manifest_sha256 = None
    if args.indices_json is not None:
        raw_manifest = open(args.indices_json, 'rb').read()
        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        manifest = json.loads(raw_manifest.decode('utf-8'))
        selected_indices = manifest.get('indices', manifest) if isinstance(manifest, dict) else manifest
        selected_indices = [int(index) for index in selected_indices]
        split_index_set = set(full_indices)
        invalid_indices = [index for index in selected_indices if index not in split_index_set]
        if invalid_indices:
            raise ValueError(f'Manifest contains indices outside split={split_name}: {invalid_indices[:10]}')
        eval_subset = Subset(ds, selected_indices)
        if is_main:
            print(f'  Fixed manifest: {len(selected_indices)} entries, SHA-256={manifest_sha256}')
    elif cfg.evaluate.subset_ratio < 1.0:
        n_keep = max(1, int(total_n * cfg.evaluate.subset_ratio))
        sel = random.Random(seed).sample(full_indices, n_keep)
        if is_main:
            print(f'  Random subset: {n_keep}/{total_n}')
        eval_subset = Subset(ds, sel)
    else:
        if is_main:
            print(f'  Full {split_name} split: {total_n} entries')
        eval_subset = full_subset
    global_eval_count = len(eval_subset)
    if distributed:
        local_indices = list(range(rank, global_eval_count, world_size))
        eval_subset = Subset(eval_subset, local_indices)
        if is_main:
            print(f'  Distributed shard: {global_eval_count} total entries; rank 0 handles {len(eval_subset)}')

    zmol_target_cache = None
    if condition_source == 'zmol':
        from utils.dataset import _cache_paths
        zmol_path = _cache_paths(getattr(cfg.dataset, 'cache_dir', './data/cache'))['zmol']
        if not os.path.exists(zmol_path):
            raise FileNotFoundError(f'Oracle evaluation requires a Zmol cache: {zmol_path}')
        try:
            zmol_target_cache = torch.load(zmol_path, weights_only=False)
        except TypeError:
            zmol_target_cache = torch.load(zmol_path)
        if is_main:
            print(f'  Oracle Zmol cache: {len(zmol_target_cache)} entries ({zmol_path})')
    collate_fn = make_msg_diffms_collate_with_cache(
        zms_cache, zmol_target_cache=zmol_target_cache
    )
    loader = DataLoader(eval_subset, batch_size=cfg.evaluate.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_fn)

    # [3/3] model
    if is_main:
        print('\n[3/3] Loading the MS2Mol checkpoint...')
    model = FLASH(
        cfg.model,
        num_node_types=len(cfg.chem.atomic_numbers) + 1,
        num_edge_types=len(cfg.chem.mol_bond_types) + 1,
    ).to(device)
    try:
        sd = torch.load(ms2mol_ckpt, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(ms2mol_ckpt, map_location=device)
    state = sd.get('model', sd)
    lora_keys = [key for key in state if '.lora_A' in key or '.lora_B' in key]
    if lora_keys:
        raise RuntimeError(
            'This release does not support LoRA checkpoints; '
            f'detected {len(lora_keys)} LoRA parameter keys.'
        )
    has_adapter_weights = any(k.startswith('zms_adapter.') for k in state.keys())
    miss, unex = model.load_state_dict(state, strict=False)
    expected_adapter_missing = {
        'zms_adapter.0.weight', 'zms_adapter.0.bias',
        'zms_adapter.3.weight', 'zms_adapter.3.bias',
    } if not has_adapter_weights else set()
    if set(miss) != expected_adapter_missing or unex:
        raise RuntimeError(
            'The MS2Mol checkpoint is incompatible with the inference model: '
            f'missing={sorted(miss)}, unexpected={sorted(unex)}'
        )
    decoder_sha256_before_adapter = non_adapter_state_sha256(model)
    if is_main:
        print(f'  ckpt: missing={len(miss)}, unexpected={len(unex)}')


    adapter_source = ms2mol_ckpt if has_adapter_weights else None
    if args.adapter_ckpt is not None:
        try:
            adapter_payload = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
        except TypeError:
            adapter_payload = torch.load(args.adapter_ckpt, map_location='cpu')
        adapter_state_full = adapter_payload.get('model', adapter_payload)
        adapter_state = {
            key[len('zms_adapter.'):]: value
            for key, value in adapter_state_full.items()
            if key.startswith('zms_adapter.')
        }
        expected_adapter_keys = set(model.zms_adapter.state_dict())
        if set(adapter_state) != expected_adapter_keys:
            raise RuntimeError(
                'adapter-only state-dict keys are incomplete: '
                f'expected={sorted(expected_adapter_keys)}, actual={sorted(adapter_state)}'
            )
        model.zms_adapter.load_state_dict(adapter_state, strict=True)
        adapter_source = args.adapter_ckpt
        has_adapter_weights = True
        decoder_sha256_after_adapter = non_adapter_state_sha256(model)
        if decoder_sha256_after_adapter != decoder_sha256_before_adapter:
            raise RuntimeError('Loading the adapter changed non-adapter decoder parameters')
        if is_main:
            print(f'  Adapter-only overlay: {args.adapter_ckpt}')
            print(f'  Decoder SHA-256 unchanged: {decoder_sha256_after_adapter}')
    else:
        decoder_sha256_after_adapter = decoder_sha256_before_adapter

    if condition_source == 'zmol':
        model.use_zms_adapter = False
        if is_main:
            print('  Oracle mode: real Zmol input; adapter disabled')
    elif has_adapter_weights and hasattr(model, 'zms_adapter') and model.zms_adapter is not None:
        model.use_zms_adapter = True
        if is_main:
            print('  Checkpoint contains zms_adapter weights; adapter enabled')
    else:
        model.use_zms_adapter = False
        if is_main:
            print('  Checkpoint has no zms_adapter weights; using Zms directly')

    model.eval()


    topk_list = list(cfg.evaluate.topk)
    n_samples = int(cfg.evaluate.n_samples)
    n_timesteps = int(cfg.evaluate.n_timesteps)
    inner_chunk = int(cfg.evaluate.inner_chunk)
    if inner_chunk > n_samples:
        inner_chunk = n_samples
    chunk_sizes = []
    remain = n_samples
    while remain > 0:
        c = min(inner_chunk, remain)
        chunk_sizes.append(c)
        remain -= c

    topk_correct = {k: 0 for k in topk_list}
    total_mols = 0
    total_edges = 0; edge_correct = 0
    bond_total = 0;  bond_correct = 0
    validity_totals = {
        'n_generated': 0,
        'n_valid': 0,
        'n_valid_connected': 0,
        'n_invalid': 0,
        'n_disconnected': 0,
    }

    tag_progress = (f'{cfg.evaluate.split}_n{int(cfg.evaluate.n_samples)}'
                    f'_T{int(cfg.evaluate.n_timesteps)}'
                    f'_r{cfg.evaluate.subset_ratio}_seed{int(cfg.evaluate.seed)}'
                    f'_{condition_source}_{condition_mode}')
    rank_suffix = f'_rank{rank:02d}of{world_size}' if distributed else ''
    progress_jsonl_path = args.progress_jsonl
    if args.resume_progress and progress_jsonl_path is None:
        out_dir_early = cfg.evaluate.output_dir
        os.makedirs(out_dir_early, exist_ok=True)
        progress_jsonl_path = os.path.join(
            out_dir_early,
            f'progress_{tag_progress}_bs{int(cfg.evaluate.batch_size)}{rank_suffix}.jsonl',
        )
    progress_fp = None
    completed_batches = set()
    if args.resume_progress:
        progress = load_compact_progress(progress_jsonl_path, topk_list)
        completed_batches = set(progress['done_batches'])
        for k in topk_list:
            topk_correct[k] = int(progress['topk_correct'][k])
        total_mols = int(progress['total_mols'])
        edge_correct = int(progress['edge_correct'])
        total_edges = int(progress['total_edges'])
        bond_correct = int(progress['bond_correct'])
        bond_total = int(progress['bond_total'])
        progress_fp = open(progress_jsonl_path, 'a', encoding='utf-8')
        print(
            f'  rank {rank}: compact progress: {progress_jsonl_path} '
            f'(completed batch={len(completed_batches)}, mols={total_mols})'
        )


    save_per_spec = (
        False if args.no_save_per_spec
        else (args.save_per_spec or cfg.evaluate.get('save_per_spec', False))
    )
    per_spec_jsonl_path = None
    per_spec_fp = None
    if save_per_spec:
        out_dir_early = cfg.evaluate.output_dir
        os.makedirs(out_dir_early, exist_ok=True)
        tag_early = tag_progress
        per_spec_jsonl_path = os.path.join(out_dir_early, f'per_spec_{tag_early}{rank_suffix}.jsonl')
        per_spec_fp = open(per_spec_jsonl_path, 'w')
        print(f'  rank {rank}: writing per-spectrum results to {per_spec_jsonl_path}')

    save_candidate_cache = (
        False if args.no_save_candidate_cache
        else (args.save_candidate_cache or cfg.evaluate.get('save_candidate_cache', False))
    )
    candidate_cache_jsonl_path = args.candidate_cache_jsonl
    candidate_cache_fp = None
    if save_candidate_cache:
        out_dir_early = cfg.evaluate.output_dir
        os.makedirs(out_dir_early, exist_ok=True)
        if candidate_cache_jsonl_path is None:
            candidate_cache_jsonl_path = os.path.join(
                out_dir_early,
                f'candidate_cache_{tag_progress}_bs{int(cfg.evaluate.batch_size)}{rank_suffix}.jsonl',
            )
        cache_mode = 'a' if args.resume_progress else 'w'
        candidate_cache_fp = open(candidate_cache_jsonl_path, cache_mode, encoding='utf-8')
        print(f'  rank {rank}: writing candidate cache to {candidate_cache_jsonl_path}')

    t_start = time.time()
    pbar = tqdm(
        loader,
        desc=f'eval {split_name}' if not distributed else f'eval {split_name} rank{rank}',
        total=len(loader),
        disable=distributed and not is_main,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            if args.resume_progress and batch_idx in completed_batches:
                postfix = {f'top{k}': f'{topk_correct[k]/max(1,total_mols)*100:.2f}%'
                           for k in topk_list}
                postfix['n_mols'] = total_mols
                postfix['resume'] = 'skip'
                pbar.set_postfix(postfix)
                continue
            batch = batch.to(device)
            bsize = batch.cond_emb_cached.size(0)
            if condition_mode == 'default':
                instrument_idx = torch.full((bsize,), 2, dtype=torch.long, device=device)
                ionization_idx = torch.zeros(bsize, dtype=torch.long, device=device)
            else:
                instrument_idx = batch.instrument_type_idx_batch.to(device) \
                    if getattr(batch, 'instrument_type_idx_batch', None) is not None \
                    else torch.zeros(bsize, dtype=torch.long, device=device)
                ionization_idx = batch.ionization_type_idx_batch.to(device) \
                    if getattr(batch, 'ionization_type_idx_batch', None) is not None \
                    else torch.zeros(bsize, dtype=torch.long, device=device)
            cond_emb_base = (
                batch.zmol_target if condition_source == 'zmol'
                else batch.cond_emb_cached
            )
            if cond_emb_base is None:
                raise RuntimeError(f'No condition vector is available for condition_source={condition_source}')

            edge_true = batch.halfedge_type
            num_nodes = batch.node_type.shape[0]
            num_edges = batch.halfedge_index.shape[1]

            preds_per_sample = []
            for chunk_n in chunk_sizes:
                node_types_exp = batch.node_type.repeat(chunk_n)
                edge_index_exp = batch.halfedge_index.repeat(1, chunk_n)
                for s in range(1, chunk_n):
                    edge_index_exp[:, s * num_edges:(s + 1) * num_edges] += s * num_nodes
                batch_node_exp = torch.cat([batch.node_type_batch + s * bsize
                                             for s in range(chunk_n)])
                batch_edge_exp = torch.cat([batch.halfedge_type_batch + s * bsize
                                             for s in range(chunk_n)])
                instrument_exp = instrument_idx.repeat(chunk_n)
                ionization_exp = ionization_idx.repeat(chunk_n)
                cond_emb_exp = cond_emb_base.repeat(chunk_n, 1)

                pred_chunk = model.sample_bfn(
                    node_types=node_types_exp,
                    edge_index=edge_index_exp,
                    batch_node=batch_node_exp,
                    batch_edge=batch_edge_exp,
                    instrument_type_idx=instrument_exp,
                    ionization_type_idx=ionization_exp,
                    cond_emb_cached=cond_emb_exp,
                    n_timesteps=n_timesteps,
                    disable_tqdm=True,
                )
                for s in range(chunk_n):
                    preds_per_sample.append(pred_chunk[s * num_edges:(s + 1) * num_edges].cpu())
                del pred_chunk
                if device != 'cpu':
                    torch.cuda.empty_cache()
            assert len(preds_per_sample) == n_samples

            edge_true_cpu = edge_true.cpu()
            pred0 = preds_per_sample[0]
            edge_correct += (pred0 == edge_true_cpu).sum().item()
            total_edges += edge_true_cpu.numel()
            bond_mask = edge_true_cpu > 0
            bond_correct += ((pred0 == edge_true_cpu) & bond_mask).sum().item()
            bond_total += bond_mask.sum().item()

            num_mols = int(batch.halfedge_type_batch.max().item()) + 1
            edge_batch_cpu = batch.halfedge_type_batch.cpu()
            node_batch_cpu = batch.node_type_batch.cpu()
            node_type_cpu = batch.node_type.cpu()
            halfedge_index_cpu = batch.halfedge_index.cpu()
            eval_mode = cfg.evaluate.get('eval_mode', 'isomorphic')
            atomic_numbers = list(cfg.chem.atomic_numbers)
            progress_records = []
            candidate_cache_records = []
            for mol_idx in range(num_mols):
                mol_mask = (edge_batch_cpu == mol_idx)
                if mol_mask.sum().item() == 0:
                    continue
                true_seq = edge_true_cpu[mol_mask]
                pred_seqs = [p[mol_mask] for p in preds_per_sample]
                mol_pred0 = pred_seqs[0]
                mol_bond_mask = true_seq > 0
                mol_edge_correct = int((mol_pred0 == true_seq).sum().item())
                mol_total_edges = int(true_seq.numel())
                mol_bond_correct = int(((mol_pred0 == true_seq) & mol_bond_mask).sum().item())
                mol_bond_total = int(mol_bond_mask.sum().item())
                spec_id = batch.mol_ids[mol_idx] if hasattr(batch, 'mol_ids') else None
                if hasattr(spec_id, 'item'):
                    spec_id = spec_id.item()
                if spec_id is not None and not isinstance(spec_id, (str, int, float, bool)):
                    spec_id = str(spec_id)
                smiles = batch.smiles[mol_idx] if hasattr(batch, 'smiles') else None

                node_mask = (node_batch_cpu == mol_idx)
                mol_node_types = node_type_cpu[node_mask]
                he_full = halfedge_index_cpu[:, mol_mask]
                node_off = node_mask.nonzero(as_tuple=True)[0][0].item()
                mol_halfedge_index = he_full - node_off
                hits = topk_hit_for_mol(
                    pred_seqs, true_seq, topk_list, mode=eval_mode,
                    node_types=mol_node_types,
                    halfedge_index=mol_halfedge_index,
                    atomic_numbers=atomic_numbers,
                    true_smiles=smiles,
                )
                cache_rec = None
                if candidate_cache_fp is not None or eval_mode == 'diffms_2d':
                    cache_rec = build_candidate_cache_record(
                        pred_seqs,
                        true_seq,
                        node_types=mol_node_types,
                        halfedge_index=mol_halfedge_index,
                        atomic_numbers=atomic_numbers,
                        true_smiles=smiles,
                    )
                    pred_stats = cache_rec['pred']
                    for key in validity_totals:
                        validity_totals[key] += int(pred_stats.get(key, 0))
                if candidate_cache_fp is not None:
                    cache_rec.update({
                        'rank': rank,
                        'world_size': world_size,
                        'batch_idx': batch_idx,
                        'mol_local_idx': mol_idx,
                        'spec_id': spec_id,
                        'eval_mode': eval_mode,
                        'hits': {str(k): bool(v) for k, v in hits.items()},
                    })
                    candidate_cache_records.append(cache_rec)
                for k in topk_list:
                    if hits[k]:
                        topk_correct[k] += 1
                total_mols += 1
                if save_per_spec and per_spec_fp is not None:

                    _rec = {
                        'mol_local_idx': mol_idx,
                        'batch_idx': batch_idx,
                        'spec_id': batch.mol_ids[mol_idx] if hasattr(batch, 'mol_ids') else None,
                        'smiles': batch.smiles[mol_idx] if hasattr(batch, 'smiles') else None,
                        'true_edge': true_seq.tolist(),
                        'pred_edges': [p.tolist() for p in pred_seqs],
                        'hits': {str(k): bool(v) for k, v in hits.items()},
                    }
                    per_spec_fp.write(json.dumps(_rec, ensure_ascii=False) + '\n')
                if args.resume_progress and progress_fp is not None:
                    progress_records.append({
                        'type': 'mol',
                        'rank': rank,
                        'world_size': world_size,
                        'batch_idx': batch_idx,
                        'mol_local_idx': mol_idx,
                        'spec_id': spec_id,
                        'smiles': smiles,
                        'hits': {str(k): bool(v) for k, v in hits.items()},
                        'edge_correct': mol_edge_correct,
                        'total_edges': mol_total_edges,
                        'bond_correct': mol_bond_correct,
                        'bond_total': mol_bond_total,
                    })

            postfix = {f'top{k}': f'{topk_correct[k]/max(1,total_mols)*100:.2f}%'
                       for k in topk_list}
            postfix['n_mols'] = total_mols
            pbar.set_postfix(postfix)

            if candidate_cache_fp is not None:
                for rec in candidate_cache_records:
                    candidate_cache_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                candidate_cache_fp.write(json.dumps({
                    'schema': 'flash_candidate_cache.v1',
                    'type': 'batch_done',
                    'rank': rank,
                    'world_size': world_size,
                    'batch_idx': batch_idx,
                    'num_mols': len(candidate_cache_records),
                }, ensure_ascii=False) + '\n')
                candidate_cache_fp.flush()

            if args.resume_progress and progress_fp is not None:
                for rec in progress_records:
                    progress_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                progress_fp.write(json.dumps({
                    'type': 'batch_done',
                    'rank': rank,
                    'world_size': world_size,
                    'batch_idx': batch_idx,
                    'num_mols': len(progress_records),
                }, ensure_ascii=False) + '\n')
                progress_fp.flush()
                completed_batches.add(batch_idx)


            if save_per_spec and per_spec_fp is not None:
                per_spec_fp.flush()

    elapsed = time.time() - t_start


    if per_spec_fp is not None:
        per_spec_fp.close()
        print(f'\nrank {rank}: per-spectrum results saved to {per_spec_jsonl_path}')
    if progress_fp is not None:
        progress_fp.close()
        print(f'\nrank {rank}: compact progress saved to {progress_jsonl_path}')
    if candidate_cache_fp is not None:
        candidate_cache_fp.close()
        print(f'\nrank {rank}: candidate cache saved to {candidate_cache_jsonl_path}')

    if distributed:
        metric_values = [topk_correct[k] for k in topk_list]
        metric_values += [total_mols, edge_correct, total_edges, bond_correct, bond_total]
        metric_values += [validity_totals[k] for k in (
            'n_generated', 'n_valid', 'n_valid_connected',
            'n_invalid', 'n_disconnected'
        )]
        metric_tensor = torch.tensor(metric_values, dtype=torch.float64, device=device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        pos = 0
        for k in topk_list:
            topk_correct[k] = int(metric_tensor[pos].item())
            pos += 1
        total_mols = int(metric_tensor[pos].item()); pos += 1
        edge_correct = int(metric_tensor[pos].item()); pos += 1
        total_edges = int(metric_tensor[pos].item()); pos += 1
        bond_correct = int(metric_tensor[pos].item()); pos += 1
        bond_total = int(metric_tensor[pos].item()); pos += 1
        for key in ('n_generated', 'n_valid', 'n_valid_connected',
                    'n_invalid', 'n_disconnected'):
            validity_totals[key] = int(metric_tensor[pos].item())
            pos += 1
        elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed = float(elapsed_tensor.item())

    if is_main:
        print('\n' + '=' * 70)
        print('DiffMS-style Top-k results')
        print('=' * 70)
        print(f'  split        : {split_name}')
        print(f'  spectra       : {total_mols}')
        print(f'  n_samples     : {n_samples}')
        print(f'  n_timesteps  : {n_timesteps}')
        print(f'  elapsed       : {elapsed:.1f}s ({elapsed/max(1,total_mols)*1000:.0f}ms/spectrum)')
        print('-' * 70)
        for k in topk_list:
            acc = topk_correct[k] / max(1, total_mols)
            print(f'  top{k:<3}       : {acc*100:.4f}%  ({topk_correct[k]}/{total_mols})')
        print('-' * 70)
        print(f'  edge_acc     : {edge_correct/max(1,total_edges)*100:.4f}%')
        print(f'  bond_acc     : {bond_correct/max(1,bond_total)*100:.4f}%')
        if validity_totals['n_generated']:
            print(f'  valid        : {validity_totals["n_valid"]/validity_totals["n_generated"]*100:.4f}%')
            print(f'  valid+conn   : {validity_totals["n_valid_connected"]/validity_totals["n_generated"]*100:.4f}%')
        print('=' * 70)

    out_dir = cfg.evaluate.output_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = (f'{split_name}_n{n_samples}_T{n_timesteps}'
           f'_r{cfg.evaluate.subset_ratio}_seed{seed}'
           f'_{condition_source}_{condition_mode}')
    if is_main:
        summary = {
            'split': split_name, 'total_mols': total_mols,
            'n_samples': n_samples, 'n_timesteps': n_timesteps,
            'subset_ratio': cfg.evaluate.subset_ratio, 'seed': seed,
            'distributed': distributed, 'world_size': world_size,
            'elapsed_sec': elapsed,
            'ms2mol_ckpt': ms2mol_ckpt, 'align_ckpt': align_ckpt,
            'adapter_ckpt': args.adapter_ckpt,
            'adapter_source': adapter_source,
            'decoder_state_sha256': decoder_sha256_after_adapter,
            'indices_json': args.indices_json,
            'indices_manifest_sha256': manifest_sha256,
            'resume_progress': bool(args.resume_progress),
            'progress_jsonl_rank0': progress_jsonl_path if args.resume_progress else None,
            'eval_mode': cfg.evaluate.get('eval_mode', 'isomorphic'),
            'condition_mode': condition_mode,
            'condition_source': condition_source,
            'candidate_cache_jsonl_rank0': candidate_cache_jsonl_path if save_candidate_cache else None,
            'topk': {f'top{k}': topk_correct[k] / max(1, total_mols) for k in topk_list},
            'topk_correct': {f'top{k}': topk_correct[k] for k in topk_list},
            'edge_accuracy': edge_correct / max(1, total_edges),
            'bond_accuracy': bond_correct / max(1, bond_total),
            'validity': {
                **validity_totals,
                'valid_rate': validity_totals['n_valid'] / max(1, validity_totals['n_generated']),
                'valid_connected_rate': validity_totals['n_valid_connected'] / max(1, validity_totals['n_generated']),
            },
        }
        summary_path = os.path.join(out_dir, f'summary_{tag}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSummary written to {summary_path}')

    if distributed:
        dist.barrier()
        dist.destroy_process_group()




if __name__ == '__main__':
    main()
