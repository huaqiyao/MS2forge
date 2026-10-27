"""Train and freeze the complete reranker from dataset-specific candidate caches."""
import argparse
import glob
import importlib.metadata
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rerank import training_common as tc
from rerank.build_candidates import build_candidates
from rerank.build_pairs import build_pairs
from rerank.prepare_labels import audit_splits, attach_labels
from utils.config import load_yaml

REPO = Path(__file__).resolve().parents[1]


def load_config(path, dataset=None):
    with Path(path).open(encoding='utf-8') as handle:
        config = load_yaml(handle, dataset=dataset)
    for key in ('data_root', 'output_dir'):
        config[key] = str(Path(config[key]).resolve())
    return config


def options(parameters, allowed):
    unknown = set(parameters) - set(allowed)
    if unknown:
        raise ValueError(f'Unsupported training parameters: {sorted(unknown)}')
    return [part for name, value in parameters.items() for part in ('--' + name.replace('_', '-'), str(value))]


def train(config):
    data, out = Path(config['data_root']), Path(config['output_dir'])
    if out == REPO or REPO in out.parents:
        raise ValueError('Training output must be outside the source repository')
    workers = int(config['workers'])
    if workers < 1:
        raise ValueError('workers must be positive')
    audit = audit_splits(data / 'labels.tsv', data / 'split.tsv')
    paths = {role: sorted(glob.glob(config['candidate_globs'][role])) for role in ('train', 'val')}
    if not all(paths.values()):
        raise ValueError('Missing train/val caches; run rerank.prepare_sampling and the generated sampling commands first')
    tc.fresh_directory(out)
    (out / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'split_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    logs = out / 'logs'
    logs.mkdir()
    commands = []

    def execute(module, argv):
        command = [sys.executable, '-m', module, *map(str, argv)]
        commands.append(command)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        log = logs / f'{len(commands):02d}_{module.rsplit(".", 1)[-1]}.log'
        print(f'[{len(commands)}] {module}', flush=True)
        env = {**os.environ, 'PYTHONHASHSEED': str(config['seed']), 'OMP_NUM_THREADS': str(workers),
               'MKL_NUM_THREADS': str(workers), 'OPENBLAS_NUM_THREADS': str(workers)}
        with log.open('w', encoding='utf-8') as handle:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{module} failed; see {log}\n{log.read_text(encoding="utf-8")[-4000:]}')

    roles = tc.split_roles(data / 'split.tsv')
    input_paths = [data / 'labels.tsv', data / 'split.tsv']
    model_dir = out / 'models'
    model_dir.mkdir()
    stats = model_dir / 'proxy_stats.json'
    for role in ('train', 'val'):
        folder = out / role
        folder.mkdir()
        summary = build_candidates(config['candidate_globs'][role], folder / 'candidates.tsv',
                                   folder / 'spectrum_status.tsv', folder / 'candidates.json')
        manifest = pd.read_csv(folder / 'spectrum_status.tsv', sep='\t', dtype=str)
        expected = {sid for sid, split in roles.items() if split == role}
        actual = set(manifest.spec_id)
        if actual != expected:
            raise ValueError(f'{role} cache coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
        if not summary['n_candidates']:
            raise ValueError(f'{role}: no valid candidates')
        active = manifest.loc[manifest.n_candidates.astype(int) > 0, 'spec_id']
        input_paths.extend(Path(p) for p in paths[role])
        for parent in sorted({Path(p).parent for p in paths[role]}):
            input_paths.extend(sorted(parent.glob('run_rank*.json')))
        input_paths.extend(data / 'spec_files' / f'{sid}.ms' for sid in active)
        execute('rerank.features', ['--candidate-structures', folder / 'candidates.tsv', '--msg-root', data,
                '--out-table', folder / 'evidence.tsv', '--summary-json', folder / 'evidence.json',
                '--accept-all-valid-rows', '--top-peaks', 30, '--ppm', 20, '--abs-tol', .01,
                '--max-single-cuts', 48, '--max-double-cut-pairs', 12, '--max-ring-open-pairs', 8,
                '--max-ion-hypotheses', 256])
        argv = ['--base-table', folder / 'candidates.tsv', '--erl-glob', folder / 'evidence.tsv',
                '--out-table', folder / 'features.tsv', '--summary-json', folder / 'features.json',
                '--data-root', data, '--workers', workers, '--stats-json', stats,
                '--declared-split', role, '--split-table', data / 'split.tsv']
        execute('rerank.assemble_features', argv + (['--fit-stats'] if role == 'train' else []))
        attach_labels(folder / 'features.tsv', data / 'labels.tsv', data / 'split.tsv', role, folder / 'training.tsv')
        execute('rerank.build_similarity_targets', ['--input-table', folder / 'training.tsv',
                '--output-targets', folder / 'targets.tsv', '--summary-json', folder / 'targets.json',
                '--split-table', data / 'split.tsv', '--declared-split', role])
    train_audit_tables = [out / 'train' / name for name in ('features.tsv', 'training.tsv', 'targets.tsv')]
    valid_audit_tables = [out / 'val' / name for name in ('features.tsv', 'training.tsv', 'targets.tsv')]

    def check_input_ids():
        result = tc.audit_input_ids(data / 'split.tsv', train_audit_tables, valid_audit_tables)
        (out / 'input_id_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

    check_input_ids()
    common = ['--train-table', out / 'train/training.tsv', '--valid-table', out / 'val/training.tsv',
              '--split-table', data / 'split.tsv', '--seed', config['seed'], '--n-jobs', workers]
    tree_keys = {'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'colsample', 'l2'}
    execute('rerank.train_evidencerank', common + ['--out-dir', out / 'exact'] + options(config['evidencerank'], tree_keys))
    execute('rerank.train_stage_a', common + ['--out-dir', out / 'stage_a',
            '--train-targets', out / 'train/targets.tsv', '--valid-targets', out / 'val/targets.tsv',
            '--features-from-metrics', out / 'exact/metrics.json', '--exact-valid-scores', out / 'exact/valid_scores.tsv']
            + options(config['stage_a'], tree_keys | {'exact_boost', 'truth_absent_group_weight', 'fusion_exact_weights',
                                                     'relevance_bins', 'similarity_exponent'}))
    with (out / 'exact/evidencerank_full_model.pkl').open('rb') as handle:
        exact = pickle.load(handle)
    source = out / 'train/features.tsv'
    score_path = out / 'train/er_scores.tsv'
    for index, chunk in enumerate(pd.read_csv(source, sep='\t', dtype={'spec_id': str, 'candidate_hash': str}, chunksize=100000)):
        result = chunk[['spec_id', 'candidate_hash']].assign(score=exact['model'].predict(chunk[exact['features']].to_numpy(np.float32)))
        result.to_csv(score_path, sep='\t', index=False, mode='w' if index == 0 else 'a', header=index == 0, float_format='%.12g')
    del exact
    build_pairs(out / 'train/training.tsv', out / 'train/targets.tsv', score_path,
                data / 'labels.tsv', data / 'split.tsv', out / 'train/pairs.tsv')
    train_audit_tables.extend([score_path, out / 'train/pairs.tsv'])
    check_input_ids()
    metadata = ['--labels', data / 'labels.tsv', '--split-table', data / 'split.tsv',
                '--spec-dir', data / 'spec_files', '--workers', workers]
    execute('rerank.ssr', ['prepare-train', '--pairs', out / 'train/pairs.tsv',
            '--out-dir', out / 'ssr_train', *metadata])
    execute('rerank.ssr', ['prepare-candidates', '--candidate-table', out / 'val/features.tsv',
            '--out-dir', out / 'ssr_val', *metadata])
    ssr_keys = {'epochs', 'batch_size', 'workers', 'embedding_dim', 'dropout', 'lr', 'weight_decay',
                'margin', 'pair_temperature', 'nce_temperature', 'nce_weight', 'regression_weight',
                'max_grad_norm', 'monitor_fraction'}
    execute('rerank.ssr', ['train', '--train-cache', out / 'ssr_train', '--out-dir', out / 'ssr_model',
            '--device', config['device'], '--seed', config['seed']] + options(config['ssr'], ssr_keys))
    execute('rerank.ssr', ['score', '--candidate-cache', out / 'ssr_val', '--checkpoint', out / 'ssr_model/best.pt',
            '--out-scores', out / 'val/ssr_scores.tsv', '--device', config['device']])
    valid_audit_tables.extend([out / 'stage_a/valid_selected_fused_scores.tsv', out / 'val/ssr_scores.tsv'])
    id_audit = check_input_ids()
    execute('rerank.ssr', ['select', '--candidate-cache', out / 'ssr_val',
            '--base-scores', out / 'stage_a/valid_selected_fused_scores.tsv', '--neural-scores', out / 'val/ssr_scores.tsv',
            '--targets', out / 'val/targets.tsv', '--out-dir', out / 'selection', '--base-weights', config['ssr_base_weights']])
    selected = json.loads((out / 'selection/selection_summary.json').read_text(encoding='utf-8'))['selected']
    if selected is None:
        raise ValueError('No SSR fusion passed validation; include weight 1 in the grid')
    for source, name in [('exact/evidencerank_full_model.pkl', 'evidencerank_full_model.pkl'),
                         ('stage_a/stage_a_model.pkl', 'stage_a_model.pkl'), ('ssr_model/best.pt', 'ssr_contrastive_tan_best.pt')]:
        shutil.copyfile(out / source, model_dir / name)
    with (model_dir / 'stage_a_model.pkl').open('rb') as handle:
        stage = pickle.load(handle)
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'rdkit', 'lightgbm', 'torch')}
    manifest = {'schema': 'ms2forge.rerank_bundle.v1', 'dataset': config['dataset'], 'feature_mode': 'direct',
                'feature_schema': 'fragment_v2_direct_241_v1', 'max_peaks': 128,
                'stage_a_exact_weight': stage['selected_exact_weight'], 'ssr_base_weight': selected['base_weight'],
                'seed': config['seed'], 'versions': versions, 'python': sys.version,
                'sha256': {p.name: tc.sha256(p) for p in model_dir.iterdir() if p.is_file()},
                'selection_split': 'val',
                'test_used_for_training': id_audit['test_used_for_training'],
                'test_used_for_selection': id_audit['test_used_for_selection'],
                'input_id_audit': id_audit}
    source_paths = sorted((REPO / 'rerank').glob('*.py')) + [REPO / 'configs/rerank_training.json']
    manifest['source_sha256'] = {str(p.relative_to(REPO)): tc.sha256(p) for p in source_paths}
    input_paths.extend(train_audit_tables + valid_audit_tables)
    tc.write_provenance(out, input_paths, {'split_audit': audit, 'input_id_audit': id_audit, 'versions': versions})
    (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Frozen model bundle: {model_dir}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', nargs='?', choices=['msg', 'nplib1'])
    p.add_argument('--config', type=Path, default=REPO / 'configs/rerank.yml')
    p.add_argument('--assets')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device')
    p.add_argument('--workers', type=int)
    args = p.parse_args()
    if args.assets:
        os.environ['MS2FORGE_ASSETS'] = args.assets
    config = load_config(args.config, args.dataset)
    for name in ('output_dir', 'device', 'workers'):
        if getattr(args, name) is not None:
            config[name] = str(args.output_dir.resolve()) if name == 'output_dir' else getattr(args, name)
    train(config)


if __name__ == '__main__':
    main()
