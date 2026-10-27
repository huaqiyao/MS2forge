"""Morgan/LightGBM feature surrogates trained on spectrum-derived targets.

Targets must be independently computed feature values, not predictions of the
surrogate being trained. All source spectra must belong to the training split.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from rerank import features as erl
from rerank import training_common as tc

SMILES = 'candidate_canonical_smiles_nostereo'
SUPPORT = 'mass_frag_v2_candidate_common_loss_support'
OBSERVED = 'mass_frag_v2_candidate_common_loss_observed'


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid training molecule: {smiles}')
    arr = np.zeros(2048, dtype=np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=False)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    return arr


def fingerprints(smiles, workers):
    if workers == 1:
        return np.stack([fingerprint(s) for s in smiles])
    with Pool(workers) as pool:
        return np.stack(list(pool.imap(fingerprint, smiles, chunksize=128)))


def parser(kind):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-table', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--data-root', type=Path, required=kind == 'loss-flags',
                   help='Contains spec_files/<spec_id>.ms; required for loss flags.')
    tc.add_data_arguments(p)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--holdout-fraction', type=float, default=0.2)
    p.add_argument('--n-estimators', type=int, default=600 if kind == 'support' else 500)
    p.add_argument('--max-rows', type=int, default=240000 if kind == 'support' else None,
                   help='Historical support recipe reads first 240000 rows. Use 0 for all rows.')
    return p


def load_training(args, kind):
    if args.workers < 1 or args.n_jobs < 1 or args.n_estimators < 1:
        raise ValueError('workers, n-jobs and n-estimators must be positive')
    if not 0 < args.holdout_fraction < 1 or (args.max_rows is not None and args.max_rows < 0):
        raise ValueError('Invalid holdout-fraction/max-rows')
    roles = tc.split_roles(args.split_table)
    tc.validate_table_role(args.train_table, roles, 'train')
    columns = ['spec_id', SMILES, SUPPORT]
    if kind == 'loss-flags':
        columns += ['mass_frag_v2_common_loss_hits', OBSERVED]
    df = pd.read_csv(args.train_table, sep='\t', usecols=columns,
                     dtype={'spec_id': str}, nrows=args.max_rows or None)
    df[SMILES] = df[SMILES].map(tc.canonical)
    protected = tc.protected_structures(args.exclude_structures)
    excluded = int(df[SMILES].isin(protected).sum())
    df = df[~df[SMILES].isin(protected)].copy()
    if df.empty:
        raise ValueError('No eligible training candidates remain')
    for column, allowed in [(SUPPORT, {0, 1, 2, 3, 4})] + ([(OBSERVED, {0, 1})] if kind == 'loss-flags' else []):
        if not df[column].isin(allowed).all():
            raise ValueError(f'Invalid labels in {column}')
    out = tc.fresh_directory(args.out_dir)
    tc.write_provenance(out, [args.train_table, args.split_table, *args.exclude_structures],
                        {'excluded_rows': excluded, 'protected_structures': len(protected),
                         'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                        if k != 'exclude_structures'},
                         'internal_holdout': 'Random structure holdout within train, not benchmark test.'})
    return df, out


def can_split(y, fraction):
    counts = pd.Series(y).value_counts()
    n_valid = math.ceil(len(y) * fraction)
    return len(counts) >= 2 and counts.min() >= 2 and min(n_valid, len(y) - n_valid) >= len(counts)


def train(kind, argv=None):
    args = parser(kind).parse_args(argv)
    df, out = load_training(args, kind)
    from lightgbm import LGBMClassifier

    common = dict(n_estimators=args.n_estimators, learning_rate=.05, num_leaves=63,
                  max_depth=10, colsample_bytree=.7, reg_lambda=2, n_jobs=args.n_jobs,
                  random_state=args.seed, verbosity=-1)
    summary = {'fingerprint': {'type': 'Morgan', 'radius': 3, 'bits': 2048, 'chirality': False},
               'internal_holdout_only': True}
    if kind == 'support':
        conflicts = df.groupby(SMILES)[SUPPORT].nunique()
        if (conflicts > 1).any():
            raise ValueError('Conflicting support labels for the same structure')
        df = df.drop_duplicates(SMILES).reset_index(drop=True)
        y = df[SUPPORT].to_numpy(np.int8)
        if not can_split(y, args.holdout_fraction):
            raise ValueError('Not enough examples per class for stratified support holdout')
        x = fingerprints(df[SMILES].tolist(), args.workers)
        train_idx, valid_idx = train_test_split(np.arange(len(y)), test_size=args.holdout_fraction,
                                               random_state=args.seed, stratify=y)
        model = LGBMClassifier(objective='multiclass', num_class=len(np.unique(y)), min_child_samples=30, **common)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[valid_idx])
        df['role'] = 'internal_holdout'
        df.loc[train_idx, 'role'] = 'train'
        df[[SMILES, SUPPORT, 'role']].to_csv(out / 'structure_split.tsv', sep='\t', index=False)
        summary.update(schema='ms2forge.massfrag_support_surrogate.v1', n_unique_structures=len(df),
                       train_rows=len(train_idx), test_rows=len(valid_idx),
                       test_accuracy=float(accuracy_score(y[valid_idx], pred)),
                       confusion_matrix=confusion_matrix(y[valid_idx], pred, labels=[0,1,2,3,4]).tolist())
        bundle = {'model': model, 'summary': summary}
        filename = 'massfrag_support_surrogate.pkl'
    else:
        single = df[df.mass_frag_v2_common_loss_hits.eq(1)].copy()
        losses = {}
        for sid in single.spec_id.unique():
            if Path(sid).name != sid or '/' in sid or '\\' in sid:
                raise ValueError(f'Invalid spectrum ID: {sid}')
            parent, _, peaks = erl.parse_ms_file(args.data_root / 'spec_files' / f'{sid}.ms')
            losses[sid] = [name for name, (loss, _) in erl.COMMON_LOSSES.items()
                           if any(abs(mz - (parent - loss)) <= max(.01, 20e-6 * (parent - loss)) for mz, _ in peaks)]
            if len(losses[sid]) != 1:
                raise ValueError(f'{sid}: single neutral loss does not match supplied feature table')
        single['loss_name'] = single.spec_id.map(lambda sid: losses[sid][0])
        labels = single[[SMILES, 'loss_name', OBSERVED]].drop_duplicates().reset_index(drop=True)
        if labels.empty:
            raise ValueError('No single-loss examples available')
        if (labels.groupby([SMILES, 'loss_name'])[OBSERVED].nunique() > 1).any():
            raise ValueError('Conflicting loss flags for the same structure/loss')
        smiles = labels[SMILES].drop_duplicates().tolist()
        x = fingerprints(smiles, args.workers)
        fp_map = {s: i for i, s in enumerate(smiles)}
        models, metrics = {}, {}
        labels['role'] = 'unused'
        for name in erl.COMMON_LOSSES:
            part = labels[labels.loss_name.eq(name)]
            y = part[OBSERVED].to_numpy(np.int8)
            if len(part) < 100 or not can_split(y, args.holdout_fraction):
                metrics[name] = {'n_rows': len(part), 'status': 'insufficient',
                                 'positive_rate': float(y.mean()) if len(y) else 0.0}
                continue
            tr, va = train_test_split(np.arange(len(y)), test_size=args.holdout_fraction,
                                      random_state=args.seed, stratify=y)
            idx = np.asarray([fp_map[s] for s in part[SMILES]])
            model = LGBMClassifier(objective='binary', min_child_samples=20,
                                   scale_pos_weight=float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), **common)
            model.fit(x[idx[tr]], y[tr])
            pred = (model.predict_proba(x[idx[va]])[:, 1] >= .5).astype(np.int8)
            models[name] = model
            labels.loc[part.index[tr], 'role'] = 'train'
            labels.loc[part.index[va], 'role'] = 'internal_holdout'
            metrics[name] = {'n_rows': len(part), 'n_positive': int(y.sum()),
                             'positive_rate': float(y.mean()), 'test_accuracy': float(accuracy_score(y[va], pred))}
        labels.to_csv(out / 'structure_split.tsv', sep='\t', index=False)
        summary.update(schema='ms2forge.massfrag_loss_flag_surrogates.v1', n_single_loss_rows=len(single),
                       n_unique_flag_labels=len(labels), metrics=metrics)
        spectra = [args.data_root / 'spec_files' / f'{sid}.ms' for sid in losses]
        (out / 'spectra_sha256.json').write_text(json.dumps({str(p): tc.sha256(p) for p in spectra}, indent=2))
        if not models:
            raise ValueError('No neutral-loss classifier could be trained; no weights published')
        bundle = {'models': models, 'summary': summary}
        filename = 'massfrag_loss_flag_models.pkl'
    with (out / filename).open('wb') as handle:
        pickle.dump(bundle, handle, pickle.HIGHEST_PROTOCOL)
    (out / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
