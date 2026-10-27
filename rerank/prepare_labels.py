"""Audit reference splits and attach supervision outside feature extraction."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rerank import training_common as tc

canonical = lru_cache(maxsize=300000)(tc.canonical)


def load_references(path):
    references = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'spec', 'smiles'} <= set(reader.fieldnames or []):
            raise ValueError('labels.tsv requires spec and smiles')
        for row in reader:
            sid = row['spec']
            if not sid or sid in references:
                raise ValueError(f'Missing/duplicate reference ID: {sid}')
            references[sid] = row['smiles']
    return references


def audit_splits(labels, split_table):
    references, roles = load_references(labels), tc.split_roles(split_table)
    structures = defaultdict(set)
    counts = Counter()
    for sid, role in roles.items():
        if role not in {'train', 'val', 'test'} or sid not in references:
            raise ValueError(f'Missing reference or unsupported split for {sid}: {role}')
        structures[canonical(references[sid])].add(role)
        counts[role] += 1
    overlap = [s for s, roles in structures.items() if len(roles) > 1]
    if overlap:
        raise ValueError(f'Reference structures overlap across splits: {len(overlap)}; examples={overlap[:3]}')
    if not counts['train'] or not counts['val']:
        raise ValueError('Nonempty train and val splits required')
    return {'identity': 'canonical_nonisomeric_smiles', 'counts': dict(counts),
            'cross_split_structures': 0, 'labels_sha256': tc.sha256(labels),
            'split_sha256': tc.sha256(split_table)}


def attach_labels(table, labels, split_table, role, output):
    if role not in {'train', 'val'}:
        raise ValueError('Training labels are restricted to train/val')
    references, roles = load_references(labels), tc.split_roles(split_table)
    tc.validate_table_role(table, roles, role)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(table).open(encoding='utf-8', newline='') as source, output.open('x', encoding='utf-8', newline='') as dest:
        reader = csv.DictReader(source, delimiter='\t')
        if {'true_canonical_smiles', 'candidate_is_true'} & set(reader.fieldnames):
            raise ValueError('Expected label-free features')
        writer = csv.DictWriter(dest, fieldnames=[*reader.fieldnames, 'true_canonical_smiles', 'candidate_is_true'], delimiter='\t')
        writer.writeheader()
        for row in reader:
            truth = canonical(references[row['spec_id']])
            row['true_canonical_smiles'] = truth
            row['candidate_is_true'] = int(canonical(row['candidate_canonical_smiles']) == truth)
            writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-table', required=True, type=Path)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--split-table', required=True, type=Path)
    p.add_argument('--declared-split', required=True, choices=['train', 'val'])
    p.add_argument('--out-table', required=True, type=Path)
    args = p.parse_args()
    audit_splits(args.labels, args.split_table)
    attach_labels(args.candidate_table, args.labels, args.split_table, args.declared_split, args.out_table)


if __name__ == '__main__':
    main()
