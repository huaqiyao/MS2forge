"""Reindex official FragHub graphs for an already-cleaned Parquet file."""
import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import load, path
from .state import atomic_json, sha256
from utils.collision_energy import ENERGY_ENCODING


def prepare_pretrain(cfg):
    import pyarrow.parquet as pq
    settings = cfg['pretrain']
    clean_file, original_file, graph_file = [path(settings[key]) for key in ('parquet', 'original_parquet', 'original_graphs')]
    digest = sha256(clean_file)
    if digest != settings['parquet_sha256']:
        raise ValueError('Clean Parquet SHA-256 does not match the configured dataset')
    output = path(settings['prepared_dir'])
    output.mkdir(parents=True, exist_ok=True)
    expected = {'parquet_sha256': digest, 'original_parquet_sha256': sha256(original_file),
                'original_graphs_sha256': sha256(graph_file), 'seed': settings['seed']}
    manifest = output / 'PREPARED.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['inputs'] != expected:
            raise ValueError('Prepared FragHub inputs changed; choose a new prepared_dir')
        for name, value in saved['files'].items():
            if sha256(output / name) != value:
                raise ValueError(f'Prepared file changed: {name}')
        return output
    for name in ('clean_smiles_dict.pt', 'splits_clean_seed42.pkl'):
        if (output / name).exists():
            raise FileExistsError(f'Incomplete prepared directory: {output}; choose a new directory')
    original = pq.read_table(original_file, columns=['smiles'])['smiles'].to_numpy()
    clean = pq.read_table(clean_file, columns=['smiles'])['smiles'].to_numpy()
    graphs = torch.load(graph_file, map_location='cpu', weights_only=False)
    positions = pd.Series(np.arange(len(clean))).groupby(pd.Series(clean), sort=False).apply(lambda x: x.to_numpy()).to_dict()
    if not set(positions).issubset(graphs):
        raise ValueError('Clean SMILES absent from the original graph dictionary')
    for smiles, value in graphs.items():
        if not (original[np.asarray(value[0], dtype=int)] == smiles).all():
            raise ValueError('Original graph row indices do not match original Parquet')
    result = {smiles: [positions[smiles]] + list(value[1:]) for smiles, value in graphs.items() if smiles in positions}
    if sum(len(value[0]) for value in result.values()) != len(clean):
        raise ValueError('Graph row coverage mismatch')
    torch.save(result, output / 'clean_smiles_dict.pt')
    smiles = list(result)
    random.Random(settings['seed']).shuffle(smiles)
    count = len(smiles)
    splits = {'train': smiles[:int(count * .9)], 'val': smiles[int(count * .9):int(count * .95)],
              'test': smiles[int(count * .95):]}
    if any(len(value) == 0 for value in splits.values()):
        raise ValueError('Insufficient molecules for 90/5/5 splitting')
    with (output / 'splits_clean_seed42.pkl').open('wb') as handle:
        pickle.dump(splits, handle)
    atomic_json(manifest, {'inputs': expected,
                          'molecules': {key: len(value) for key, value in splits.items()},
                          'spectra': {key: sum(len(result[s][0]) for s in value) for key, value in splits.items()},
                          'files': {name: sha256(output / name) for name in ('clean_smiles_dict.pt', 'splits_clean_seed42.pkl')}})
    return output


def prepare_finetune(cfg):
    """Build supervised train/val arrays from metadata and observed subformulae."""
    import csv
    import tempfile
    from .inputs import metadata, graph, spectrum_counts
    settings = cfg['datasets'][cfg['dataset']]
    root, output = path(settings['data_root']), path(settings['prepared_dir'])
    if output.exists():
        raise FileExistsError(f'Prepared directory already exists: {output}; use a new location')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, rejected, graphs, graph_ids = [], [], {}, {}
    for row in metadata(root, supervised=True):
        try:
            key, nodes, edges = graph(row['smiles'])
            if row['ionization'] not in ('[M+H]+', '[M-H]-', '[M+Na]+'):
                raise ValueError('Unsupported adduct')
        except ValueError as exc:
            rejected.append(dict(spec=row['spec'], split=row['split'], reason=str(exc)))
            continue
        if key not in graphs:
            graph_ids[key] = len(graphs)
            graphs[key] = (nodes, edges)
        rows.append(dict(row, smiles=key, graph_id=graph_ids[key]))
    if any(not any(r['split'] == s for r in rows) for s in ('train', 'val')):
        raise ValueError('Preparation requires nonempty qualified train and val splits')
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        dest = Path(temp) / 'data'
        dest.mkdir()
        n = max(len(x) for x, _ in graphs.values())
        nodes = np.zeros((len(graphs), n), np.int8)
        edges = np.zeros((len(graphs), n, n), np.uint8)
        sizes = np.zeros(len(graphs), np.int16)
        for i, (x, e) in enumerate(graphs.values()):
            sizes[i] = len(x); nodes[i, :len(x)] = x; edges[i, :len(x), :len(x)] = e
        arrays = dict(nodes=nodes, edges=edges, node_counts=sizes,
                      graph_ids=np.array([r['graph_id'] for r in rows], dtype=np.int64),
                      adduct_ids=np.zeros(len(rows), np.int64), peak_counts=np.zeros(len(rows), np.int16),
                      energy_bins=np.array([r['energy_bin'] for r in rows], dtype=np.int64))
        counts = np.lib.format.open_memmap(dest / 'counts.npy', mode='w+', dtype=np.int16, shape=(len(rows), 128, 9))
        annotation_hashes = {}
        for i, row in enumerate(rows):
            counts[i], arrays['peak_counts'][i], arrays['adduct_ids'][i] = spectrum_counts(root, row['spec'], row['ionization'])
            annotation_hashes[row['spec']] = sha256(root / 'subformulae/default_subformulae' / (row['spec'] + '.json'))
        counts.flush(); del counts
        for name, value in arrays.items(): np.save(dest / (name + '.npy'), value)
        for split in ('train', 'val'): np.save(dest / (split + '_indices.npy'), np.array([i for i,r in enumerate(rows) if r['split'] == split], dtype=np.int64))
        with (dest / 'manifest.tsv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, delimiter='\t', fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        atomic_json(dest / 'excluded.json', rejected)
        atomic_json(dest / 'PREPARED.json', dict(schema='ms2forge.encoder_inputs.v2', energy_encoding=ENERGY_ENCODING,
                    dataset=cfg['dataset'], labels_sha256=sha256(root / 'labels.tsv'),
                    split_sha256=sha256(root / 'split.tsv'), subformula_sha256=annotation_hashes,
                    rows=len(rows), excluded=len(rejected), test_graphs_read=False,
                    files={p.name: sha256(p) for p in dest.iterdir() if p.is_file()}))
        dest.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/encoder.yml')
    parser.add_argument('--stage', choices=['pretrain', 'finetune'], default='finetune')
    parser.add_argument('--dataset', choices=['msg', 'nplib1'])
    args = parser.parse_args()
    cfg = load(args.config, dataset=args.dataset)
    print(prepare_pretrain(cfg) if args.stage == 'pretrain' else prepare_finetune(cfg))


if __name__ == '__main__':
    main()
