"""Shared FragHub filtering with joint MSG/NPLIB1 held-out references.

Dependencies: Python 3.10+, RDKit, pandas and pyarrow. Use RDKit 2024.09.4
to match the manuscript's molecular identity and scaffold definitions.
Input Parquet must contain a SMILES column. Benchmark labels are TSV files
with 'spec' and 'smiles'; split TSVs contain 'name' and 'split'. The aliases
'spec' for split identifiers and 'validation' for 'val' are also accepted.

All retained Parquet columns, spectrum rows and their order are preserved.
Canonical isomeric SMILES are used to count unique molecules, never to
deduplicate spectrum rows. There is no salt stripping, neutralization or
explicit tautomer canonicalization. Reports contain no date/time metadata.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


METRICS = ('full_inchikey', 'inchikey14', 'murcko_scaffold')
GROUPS = ('msg_val', 'msg_test', 'nplib1_val', 'nplib1_test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def same_value(left, right):
    """Compare nested spectrum values while treating two NaNs as equal."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_value(a, b) for a, b in zip(left, right))
    return left == right


def tables_equal(left, right):
    if (left.num_rows != right.num_rows
            or not left.schema.equals(right.schema, check_metadata=True)):
        return False
    if left.equals(right, check_metadata=True):
        return True
    # Arrow equality can report unequal NaNs; inspect logical values in bounded batches.
    for start in range(0, left.num_rows, 4096):
        if not same_value(left.slice(start, 4096).to_pylist(),
                          right.slice(start, 4096).to_pylist()):
            return False
    return True


def describe(smiles):
    """Derive the same three structure descriptors for source and references."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Empty or non-string SMILES')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f'Canonical SMILES reparse failed: {smiles!r}')
    key = Chem.MolToInchiKey(mol)
    if not key or len(key) != 27:
        raise ValueError(f'Cannot compute full Standard InChIKey: {smiles!r}')
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = (Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)
                if core.GetNumAtoms() else '')
    return {
        'canonical_smiles': canonical,
        'full_inchikey': key,
        'inchikey14': key[:14],
        'murcko_scaffold': scaffold,
    }


def load_heldout(labels_path, split_path, benchmark):
    labels = pd.read_csv(labels_path, sep='\t', dtype=str, keep_default_na=False)
    splits = pd.read_csv(split_path, sep='\t', dtype=str, keep_default_na=False)
    if not {'spec', 'smiles'} <= set(labels.columns):
        raise ValueError(f'{labels_path}: required columns are spec and smiles')
    identifier = 'name' if 'name' in splits.columns else 'spec'
    if not {identifier, 'split'} <= set(splits.columns):
        raise ValueError(f'{split_path}: required columns are name/spec and split')
    labels = labels[['spec', 'smiles']].copy()
    splits = splits[[identifier, 'split']].rename(columns={identifier: 'spec'})
    if (labels.spec.eq('').any() or splits.spec.eq('').any()
            or not labels.spec.is_unique or not splits.spec.is_unique):
        raise ValueError(f'{benchmark}: missing or duplicate spectrum identifiers')
    splits['split'] = splits['split'].replace({'validation': 'val'})
    if not splits['split'].isin(['train', 'val', 'test']).all():
        raise ValueError(f'{benchmark}: unknown split assignment')
    merged = splits.merge(labels, on='spec', how='left', validate='one_to_one',
                          indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError(f'{benchmark}: split identifiers missing from labels')
    heldout = merged[merged['split'].isin(['val', 'test'])].copy()
    for split in ('val', 'test'):
        if not heldout['split'].eq(split).any():
            raise ValueError(f'{benchmark}: empty {split} reference partition')
    heldout['group'] = benchmark + '_' + heldout['split']
    return heldout[['group', 'spec', 'smiles']].to_dict('records')


def descriptor_keys(smiles, descriptors):
    return {metric: {descriptors[s][metric] for s in smiles
                     if descriptors[s][metric]} for metric in METRICS}


def overlap_summary(source_smiles, group_rows, descriptors):
    """Count held-out canonical molecules matching a given source corpus."""
    keys = descriptor_keys(source_smiles, descriptors)
    result = {}
    for group in GROUPS:
        rows = [row for row in group_rows if row['group'] == group]
        molecules = {descriptors[row['smiles']]['canonical_smiles']:
                     descriptors[row['smiles']] for row in rows}
        result[group] = {
            'reference_spectrum_rows': len(rows),
            'reference_unique_canonical_molecules': len(molecules),
            'overlap_molecules': {
                metric: sum(bool(d[metric]) and d[metric] in keys[metric]
                            for d in molecules.values())
                for metric in METRICS
            },
        }
    return result


def parser_for(mode):
    criterion = ('full Standard InChIKey OR InChIKey-14'
                 if mode == 'molecules' else 'nonempty nonisomeric Bemis-Murcko scaffold')
    parser = argparse.ArgumentParser(
        description=f'Jointly exclude MSG/NPLIB1 val/test matches by {criterion}.',
        epilog='Both benchmark references are required. No training partitions are added '
               'to the exclusion reference. Existing output files are never overwritten.')
    parser.add_argument('--input', type=Path, required=True, help='Source FragHub Parquet')
    parser.add_argument('--output', type=Path, required=True, help='Filtered Parquet')
    parser.add_argument('--msg-labels', type=Path, required=True)
    parser.add_argument('--msg-split', type=Path, required=True)
    parser.add_argument('--nplib1-labels', type=Path, required=True)
    parser.add_argument('--nplib1-split', type=Path, required=True)
    parser.add_argument('--smiles-column', default='smiles')
    parser.add_argument('--audit', type=Path,
                        help='JSON audit output; defaults to OUTPUT.audit.json')
    return parser


def run(mode, argv=None):
    if mode not in ('molecules', 'scaffolds'):
        raise ValueError(f'Unknown cleaning mode: {mode}')
    args = parser_for(mode).parse_args(argv)
    inputs = {
        'fraghub': args.input.resolve(),
        'msg_labels': args.msg_labels.resolve(),
        'msg_split': args.msg_split.resolve(),
        'nplib1_labels': args.nplib1_labels.resolve(),
        'nplib1_split': args.nplib1_split.resolve(),
    }
    output = args.output.resolve()
    audit = (args.audit or args.output.with_suffix('.audit.json')).resolve()
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == audit or output in inputs.values() or audit in inputs.values():
        raise ValueError('Input, filtered output and audit paths must be distinct')
    for path in (output, audit):
        if path.exists():
            raise FileExistsError(f'Choose a new output path: {path}')

    source_hashes = {name: sha256(path) for name, path in inputs.items()}
    references = load_heldout(inputs['msg_labels'], inputs['msg_split'], 'msg')
    references += load_heldout(inputs['nplib1_labels'], inputs['nplib1_split'], 'nplib1')
    table = pq.read_table(inputs['fraghub'])
    if args.smiles_column not in table.column_names:
        raise ValueError(f'Parquet has no {args.smiles_column!r} column')
    source_smiles = table[args.smiles_column].to_pylist()
    if not source_smiles:
        raise ValueError('Source FragHub Parquet is empty')
    all_smiles = source_smiles + [row['smiles'] for row in references]
    if any(not isinstance(s, str) or not s.strip() for s in all_smiles):
        raise ValueError('Source or reference contains null/empty/non-string SMILES')
    descriptors = {s: describe(s) for s in sorted(set(all_smiles))}
    blocklist = descriptor_keys([r['smiles'] for r in references], descriptors)
    active = ('full_inchikey', 'inchikey14') if mode == 'molecules' else ('murcko_scaffold',)
    reasons = {
        s: [metric for metric in active if descriptors[s][metric]
            and descriptors[s][metric] in blocklist[metric]]
        for s in set(source_smiles)
    }
    keep = [not reasons[s] for s in source_smiles]
    filtered = table.filter(pa.array(keep, type=pa.bool_()))
    kept_smiles = [s for s, retained in zip(source_smiles, keep) if retained]
    removed_smiles = [s for s, retained in zip(source_smiles, keep) if not retained]
    before = overlap_summary(source_smiles, references, descriptors)
    after = overlap_summary(kept_smiles, references, descriptors)
    group_keys = {
        group: descriptor_keys([r['smiles'] for r in references if r['group'] == group],
                               descriptors)
        for group in GROUPS
    }
    for group in GROUPS:
        if any(after[group]['overlap_molecules'][metric] for metric in active):
            raise RuntimeError(f'Unexpected residual overlap in {group}')

    def counts(strings):
        return {
            'spectrum_rows': len(strings),
            'unique_canonical_molecules': len({descriptors[s]['canonical_smiles']
                                               for s in strings}),
        }

    report = {
        'status': 'PASS',
        'mode': mode,
        'rdkit_version': rdBase.rdkitVersion,
        'scope': list(GROUPS),
        'active_exclusion_metrics': list(active),
        'normalization': 'Sanitize; canonical isomeric SMILES; reparse; Standard '
                         'InChIKey and its first 14 characters; nonisomeric Murcko '
                         'scaffold. No salt stripping, neutralization or explicit '
                         'tautomer canonicalization. Empty scaffolds never match.',
        'inputs': {name: {'path': str(path), 'sha256': source_hashes[name]}
                   for name, path in inputs.items()},
        'counts': {'input': counts(source_smiles), 'removed': counts(removed_smiles),
                   'retained': counts(kept_smiles)},
        'removed_rows_by_metric': {
            metric: sum(metric in reasons[s] for s in source_smiles) for metric in active
        },
        'metric_count_note': 'Rows matching multiple active metrics are removed once; '
                             'per-metric counts must not be summed.',
        'overlap_before': before,
        'overlap_after': after,
        'retained_source_rows_zero_based': [i for i, retained in enumerate(keep) if retained],
        'removed_records': [
            {'source_row_zero_based': i, 'smiles': s, **descriptors[s],
             'matched_metrics': reasons[s],
             'matched_groups': [
                 group for group in GROUPS
                 if any(descriptors[s][metric] in group_keys[group][metric]
                        for metric in reasons[s])
             ]}
            for i, s in enumerate(source_smiles) if reasons[s]
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fraghub_', dir=output.parent) as temp:
        staged = Path(temp) / 'filtered.parquet'
        pq.write_table(filtered, staged, compression='snappy')
        reread = pq.read_table(staged)
        if not tables_equal(reread, filtered):
            raise RuntimeError('Written Parquet differs from expected retained rows')
        if any(sha256(path) != source_hashes[name] for name, path in inputs.items()):
            raise RuntimeError('An input changed during filtering')
        report['all_columns_values_and_retained_order_preserved'] = True
        report['sources_unchanged'] = True
        report['output'] = {'path': str(output), 'bytes': staged.stat().st_size,
                            'sha256': sha256(staged)}
        # Exclusive creation protects existing results, including concurrent runs.
        with audit.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        try:
            # Hard-link the verified staged file on the same filesystem, without overwrite.
            os.link(staged, output)
        except BaseException:
            audit.unlink()
            raise
    print(json.dumps({'mode': mode, 'counts': report['counts'],
                      'output': str(output), 'audit': str(audit)}, ensure_ascii=False, indent=2))
    return report
