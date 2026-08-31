"""Recompute exact-match metrics from FLASH candidate-cache JSONL files.

This script is intentionally CPU-only. It reads the compact candidate identity
cache written by scripts/sample.py and recomputes top-k metrics without running
the generative sampler again.
"""
import argparse
import glob
import json
import os
from collections import defaultdict


MODE_TO_FIELDS = {
    'diffms_inchi': ('inchi', 'diffms_inchi_counts'),
    'connected_inchikey': ('inchikey', 'connected_inchikey_counts'),
    'valid_inchikey': ('inchikey', 'valid_inchikey_counts'),
    'connected_smiles': ('canonical_smiles', 'connected_smiles_counts'),
    'valid_smiles': ('canonical_smiles', 'valid_smiles_counts'),
}


def parse_topk(text):
    return [int(x) for x in text.split(',') if x.strip()]


def load_records(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f'No cache files matched: {patterns}')

    records = {}
    done_batches = set()
    saw_batch_markers = False
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_type = rec.get('type')
                rank = int(rec.get('rank', 0))
                batch_idx = rec.get('batch_idx')
                if batch_idx is None:
                    continue
                batch_idx = int(batch_idx)
                if rec_type == 'batch_done':
                    saw_batch_markers = True
                    done_batches.add((rank, batch_idx))
                    continue
                if rec_type != 'mol_candidates':
                    continue
                mol_idx = int(rec.get('mol_local_idx', -1))
                if mol_idx < 0:
                    continue
                records[(rank, batch_idx, mol_idx)] = rec

    if saw_batch_markers:
        records = {
            key: rec
            for key, rec in records.items()
            if (key[0], key[1]) in done_batches
        }
    return paths, records


def compute_mode(records, mode, topk_list):
    if mode not in MODE_TO_FIELDS:
        raise ValueError(f'Unknown mode {mode!r}; choices: {sorted(MODE_TO_FIELDS)}')
    true_field, pred_field = MODE_TO_FIELDS[mode]
    topk_correct = {k: 0 for k in topk_list}
    total = 0
    missing_true = 0
    empty_pred = 0
    stats = defaultdict(int)

    for rec in records.values():
        true_id = rec.get('true', {}).get(true_field)
        pred_counts = rec.get('pred', {}).get(pred_field, [])
        pred_ids = [item[0] for item in pred_counts]
        if not true_id:
            missing_true += 1
            continue
        if not pred_ids:
            empty_pred += 1
        total += 1
        pred_stats = rec.get('pred', {})
        for name in ('n_generated', 'n_valid', 'n_valid_connected',
                     'n_invalid', 'n_disconnected'):
            stats[name] += int(pred_stats.get(name, 0))
        for k in topk_list:
            if true_id in pred_ids[:k]:
                topk_correct[k] += 1

    return {
        'mode': mode,
        'total_mols': total,
        'missing_true': missing_true,
        'empty_pred': empty_pred,
        'topk': {f'top{k}': topk_correct[k] / max(1, total) for k in topk_list},
        'topk_correct': {f'top{k}': topk_correct[k] for k in topk_list},
        'candidate_stats': {
            name: stats[name] / max(1, total)
            for name in ('n_generated', 'n_valid', 'n_valid_connected',
                         'n_invalid', 'n_disconnected')
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cache_glob', nargs='+',
                   help='One or more candidate_cache_*.jsonl paths/globs')
    p.add_argument('--mode', default='diffms_inchi',
                   choices=sorted(MODE_TO_FIELDS))
    p.add_argument('--topk', default='1,5,10')
    p.add_argument('--output_json', default=None)
    args = p.parse_args()

    topk_list = parse_topk(args.topk)
    paths, records = load_records(args.cache_glob)
    result = compute_mode(records, args.mode, topk_list)
    result['cache_files'] = paths

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
