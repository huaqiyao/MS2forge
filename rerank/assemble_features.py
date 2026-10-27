"""Assemble label-free features using deterministic fragment rules."""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rerank.contracts import validate_model_features, validate_feature_table
from rerank.mass_features import mass_row, fit_stats, proxy_scores
from rerank.ssr import parallel_map
from rerank import training_common as tc

def group_relative(frame, base, suffix):
    values=pd.to_numeric(frame[base],errors='coerce').fillna(0.0).astype(float)
    grouped=values.groupby(frame['spec_id'],sort=False)
    if suffix=='z':
        mean=grouped.transform('mean'); second=(values*values).groupby(frame['spec_id'],sort=False).transform('mean')
        std=np.sqrt(np.maximum(second-mean*mean,0.0)); return (values-mean)/std.where(std>1e-12,1.0)
    if suffix=='rank':
        return grouped.rank(method='first',ascending=False).astype(float)
    if suffix=='gap_top1':
        return grouped.transform('max')-values
    if suffix=='percentile':
        rank=grouped.rank(method='first',ascending=False); count=grouped.transform('size')
        return (1.0-(rank-1.0)/(count-1.0).where(count>1,1.0)).where(count>1,1.0)
    if suffix=='minmax':
        minimum=grouped.transform('min'); maximum=grouped.transform('max'); width=maximum-minimum
        return ((values-minimum)/width.where(width>1e-12,1.0)).where(width>1e-12,0.0)
    raise ValueError(suffix)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-table', required=True, type=Path)
    parser.add_argument('--erl-glob', required=True)
    parser.add_argument('--out-table', required=True, type=Path)
    parser.add_argument('--out-scores', type=Path)
    parser.add_argument('--summary-json', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--stats-json', type=Path)
    parser.add_argument('--fit-stats', action='store_true')
    parser.add_argument('--split-table', type=Path)
    parser.add_argument('--declared-split', choices=['train', 'val', 'test'], default='test')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    stats_path = args.stats_json or (args.model_dir / 'proxy_stats.json' if args.model_dir else None)
    if stats_path is None:
        parser.error('--stats-json or --model-dir is required')
    if args.fit_stats:
        if args.declared_split != 'train' or not args.split_table:
            parser.error('--fit-stats requires --declared-split train and --split-table')
        if stats_path.exists():
            parser.error('Refusing to overwrite frozen proxy statistics')
        tc.validate_table_role(args.base_table, tc.split_roles(args.split_table), 'train')
    elif not stats_path.is_file():
        parser.error(f'Missing train-fitted proxy statistics: {stats_path}')
    base = pd.read_csv(args.base_table, sep='\t', keep_default_na=False,
                       dtype={'spec_id': str, 'candidate_hash': str}, low_memory=False)
    if base.empty or base.duplicated(['spec_id', 'candidate_hash']).any():
        raise ValueError('Empty candidate table or duplicate candidate keys')
    columns = [c for c in base if any(t in c.lower() for t in ('true', 'truth', 'target', 'tanimoto_to_', 'mces_to_'))]
    if columns:
        raise ValueError(f'Feature assembly requires a label-free base table: {columns}')
    work = [(row.spec_id, row.candidate_canonical_smiles, str(args.data_root)) for row in base.itertuples()]
    mass = pd.DataFrame(parallel_map(mass_row, work, args.workers, chunksize=32))
    for column in mass:
        base[column] = mass[column].to_numpy()
    if args.fit_stats:
        stats = fit_stats(base)
        stats['source_sha256'] = tc.sha256(args.base_table)
        stats['split_sha256'] = tc.sha256(args.split_table)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    else:
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
    base['score_fragment_proxy_v2'] = proxy_scores(base, stats)
    base['score_mass_fragment_proxy_v2'] = base['score_fragment_proxy_v2']
    base['log_freq'] = np.log1p(base.candidate_freq.astype(float))
    base['inv_freq_rank'] = 1 / base.candidate_rank_freq.astype(float)
    base['freq_rank'] = base.candidate_rank_freq.astype(float)
    paths = sorted(glob.glob(args.erl_glob))
    if not paths:
        raise ValueError('No EvidenceRank feature files match --erl-glob')
    evidence = pd.concat([pd.read_csv(p, sep='\t', dtype={'spec_id': str, 'candidate_hash': str}) for p in paths], ignore_index=True)
    erl_columns = [c for c in evidence if c.startswith('erl_')]
    base = base.merge(evidence[['spec_id', 'candidate_hash', *erl_columns]],
                      on=['spec_id', 'candidate_hash'], how='left', validate='one_to_one')
    if len(base) != len(evidence) or base[erl_columns].isna().any().any():
        raise ValueError('Incomplete EvidenceRank feature join')
    features = tc.recipe('evidencerank')['features']
    validate_model_features(features)
    derived = {}
    for feature in features:
        if feature in base:
            continue
        for suffix in ('gap_top1', 'percentile', 'minmax', 'rank', 'z'):
            marker = '_group_' + suffix
            if feature.endswith(marker):
                derived[feature] = group_relative(base, feature[:-len(marker)], suffix)
                break
        else:
            raise ValueError(f'Missing feature: {feature}')
    base = pd.concat([base, pd.DataFrame(derived, index=base.index)], axis=1)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out_table, sep='\t', index=False, float_format='%.12g')
    validate_feature_table(args.out_table, features)
    if args.out_scores:
        if not args.model_dir:
            parser.error('--out-scores requires --model-dir')
        from rerank.numpy_pickle_compat import install_numpy2_pickle_aliases
        install_numpy2_pickle_aliases()
        with (args.model_dir / 'evidencerank_full_model.pkl').open('rb') as handle:
            bundle = pickle.load(handle)
        matrix = base[bundle['features']].to_numpy(np.float32)
        args.out_scores.parent.mkdir(parents=True, exist_ok=True)
        base[['spec_id', 'candidate_hash']].assign(score=bundle['model'].predict(matrix)).to_csv(
            args.out_scores, sep='\t', index=False, float_format='%.10g')
    summary = {'schema': 'ms2forge.direct_full_features.v1', 'feature_mode': 'direct',
               'n_candidates': len(base), 'n_features': len(features), 'truth_columns_read': [],
               'stats_sha256': tc.sha256(stats_path), 'input_sha256': tc.sha256(args.base_table),
               'feature_table_sha256': tc.sha256(args.out_table), 'fitted_statistics': args.fit_stats}
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
