"""Prepare per-split sampling inputs and configurations for reranker training."""
import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from rerank.train import REPO, load_config
from utils.config import load_yaml
from rerank.training_common import split_roles


def generation_inputs(args):
    ids = {sid for sid, role in split_roles(args.split_table).items() if role == args.split}
    seen = set()
    fields = ['sample_id', 'formula', 'formula_kind', 'formula_source', 'instrument', 'adduct', 'split']
    with Path(args.labels).open(encoding='utf-8-sig', newline='') as source, Path(args.output).open('x', encoding='utf-8', newline='') as dest:
        reader = csv.DictReader(source, delimiter='\t')
        writer = csv.DictWriter(dest, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in reader:
            sid = row['spec']
            if sid not in ids:
                continue
            if sid in seen:
                raise ValueError(f'Duplicate spectrum metadata: {sid}')
            seen.add(sid)
            writer.writerow(dict(sample_id=sid, formula=row['formula'], formula_kind='neutral',
                                 formula_source='provided_metadata', instrument=row['instrument'],
                                 adduct=row['ionization'], split=args.split))
    if seen != ids:
        raise ValueError(f'Missing split metadata: {len(ids-seen)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', choices=['msg', 'nplib1'])
    parser.add_argument('--config', type=Path, default=REPO / 'configs/rerank.yml')
    parser.add_argument('--assets')
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    if args.assets:
        os.environ['MS2FORGE_ASSETS'] = args.assets
    cfg = load_config(args.config, args.dataset)
    with Path(cfg['sampling_config']).open(encoding='utf-8') as handle:
        sample = load_yaml(handle, dataset=cfg['dataset'])
    data = Path(cfg['data_root'])
    commands = []
    for role, pattern in cfg['candidate_globs'].items():
        output = Path(pattern).parent
        output.mkdir(parents=True, exist_ok=True)
        inputs = output / 'generation_inputs.tsv'
        generation_inputs(SimpleNamespace(labels=data / 'labels.tsv', split_table=data / 'split.tsv',
                                          split=role, output=inputs))
        config_path = output / 'sample.yml'
        current = {**sample, 'inputs': str(inputs.resolve()), 'output': str(output.resolve()), 'seed': cfg['seed']}
        config_path.write_text(yaml.safe_dump(current, sort_keys=False), encoding='utf-8')
        command = [sys.executable, str(REPO / 'sample.py'), cfg['dataset'], '--config', str(config_path), '--gpus', str(args.gpus)]
        commands.append(command)
        print(subprocess_command(command), flush=True)
    destination = Path(cfg['output_dir']).parent / 'rerank_sampling_commands.json'
    destination.write_text(json.dumps(commands, indent=2), encoding='utf-8')


def subprocess_command(command):
    if os.name == 'nt':
        return '& ' + ' '.join("'" + item.replace("'", "''") + "'" for item in command)
    return shlex.join(command)


if __name__ == '__main__':
    main()
