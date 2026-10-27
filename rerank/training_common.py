"""Input validation and provenance shared by the reranker trainers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from rerank.contracts import validate_model_features


def recipe(name):
    path = Path(__file__).resolve().parents[1] / 'configs' / 'rerank_training.json'
    return json.loads(path.read_text(encoding='utf-8'))[name]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_data_arguments(parser):
    parser.add_argument('--split-table', required=True,
                        help='TSV with name, split; train/val/validation/test roles are checked.')
    parser.add_argument('--exclude-structures', action='append', default=[],
                        help='Repeat for MSG and NPLIB1 protected-structure TSVs with a smiles column.')


def split_roles(path):
    roles = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'name', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Split table requires name and split columns')
        for row in reader:
            sid = row['name']
            if not sid or sid in roles:
                raise ValueError(f'Missing/duplicate split ID: {sid}')
            role = row['split'].lower()
            roles[sid] = 'val' if role in {'valid', 'validation'} else role
    return roles


def canonical(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid molecule: {smiles}')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f'Multiple molecular components: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def protected_structures(paths):
    result = set()
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if 'smiles' not in (reader.fieldnames or []):
                raise ValueError(f'{path}: protected table needs smiles column')
            result.update(canonical(row['smiles']) for row in reader)
    return result


def validate_table_role(path, roles, role, protected=()):
    seen = set()
    count = 0
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if 'spec_id' not in (reader.fieldnames or []):
            raise ValueError(f'{path}: missing spec_id')
        for row in reader:
            count += 1
            sid = row['spec_id']
            if roles.get(sid) != role:
                raise ValueError(f'{path}: spectrum {sid} is not {role} in split table')
            if 'candidate_hash' in row:
                key = sid, row['candidate_hash']
                if not key[1] or key in seen:
                    raise ValueError(f'{path}: missing/duplicate candidate key {key}')
                seen.add(key)
            if protected:
                smi = row.get('candidate_canonical_smiles_nostereo', row.get('candidate_canonical_smiles'))
                if smi is None:
                    raise ValueError('Protected-structure check requires candidate SMILES')
                if canonical(smi) in protected:
                    raise ValueError(f'{path}: protected structure in {role} candidates; rebuild the table')
    if not count:
        raise ValueError(f'{path}: empty training/validation table')
    return count


def audit_input_ids(split_table, train_tables, valid_tables):
    """Check supplied input-table IDs against the declared held-out split.

    Tables may contain repeated IDs for different candidates. This audit covers
    input spectra, not molecular identity or upstream model training data.
    """
    roles = split_roles(split_table)
    test_ids = {sid for sid, role in roles.items() if role == 'test'}
    if not test_ids:
        raise ValueError('ID audit requires a nonempty test split')
    tables = {}
    used = {'train': set(), 'val': set()}
    for role, paths in (('train', train_tables), ('val', valid_tables)):
        if not paths:
            raise ValueError(f'ID audit requires {role} input tables')
        for path in paths:
            ids = set()
            count = 0
            with Path(path).open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                if 'spec_id' not in (reader.fieldnames or []):
                    raise ValueError(f'{path}: missing spec_id')
                for row in reader:
                    count += 1
                    ids.add(row['spec_id'])
            if not count:
                raise ValueError(f'{path}: empty ID audit input')
            overlap = ids & test_ids
            if overlap:
                raise ValueError(f'{path}: test IDs in {role} input: {sorted(overlap)[:5]}')
            wrong_role = {sid for sid in ids if roles.get(sid) != role}
            if wrong_role:
                raise ValueError(f'{path}: IDs outside declared {role} split: {sorted(wrong_role, key=str)[:5]}')
            used[role].update(ids)
            tables[str(Path(path).resolve())] = {
                'role': role, 'rows': count, 'unique_ids': len(ids),
                'test_overlap_count': len(overlap), 'sha256': sha256(path),
            }
    train_overlap = used['train'] & test_ids
    valid_overlap = used['val'] & test_ids
    return {
        'identity': 'spec_id',
        'split_table': str(Path(split_table).resolve()),
        'split_sha256': sha256(split_table),
        'test_id_count': len(test_ids),
        'train_input_id_count': len(used['train']),
        'validation_input_id_count': len(used['val']),
        'train_test_overlap_count': len(train_overlap),
        'validation_test_overlap_count': len(valid_overlap),
        'test_used_for_training': bool(train_overlap),
        'test_used_for_selection': bool(valid_overlap),
        'tables': tables,
    }


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory must be empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_provenance(out, inputs, extra=None):
    data = {'schema': 'ms2forge.rerank_training_inputs.v1',
            'inputs': {str(Path(p).resolve()): sha256(p) for p in inputs},
            'scope': 'Checks supplied split and exclusion lists.',
            **(extra or {})}
    (Path(out) / 'training_inputs.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


def validate_training_inputs(args):
    if getattr(args, 'test_table', None):
        raise ValueError('Training does not accept --test-table; evaluate only after freezing the model')
    roles = split_roles(args.split_table)
    protected = protected_structures(args.exclude_structures)
    validate_table_role(args.train_table, roles, 'train', protected)
    validate_table_role(args.valid_table, roles, 'val', protected)
    inputs = [args.train_table, args.valid_table, args.split_table, *args.exclude_structures]
    for key in ('train_targets', 'valid_targets', 'features_from_metrics', 'exact_valid_scores'):
        if getattr(args, key, None):
            inputs.append(getattr(args, key))
    out = fresh_directory(args.out_dir)
    write_provenance(out, inputs, {'protected_structures': len(protected)})


def validate_features(rows, features):
    validate_model_features(features)
    if not rows:
        raise ValueError('Empty candidate table')
    for row in rows:
        for name in features:
            try:
                valid = math.isfinite(float(row[name]))
            except (KeyError, ValueError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f'Missing/non-finite feature {name} for {row.get("spec_id")}')
        label = str(row.get('candidate_is_true', '')).strip().lower()
        if label not in {'true', 'false', '0', '1', 'yes', 'no'}:
            raise ValueError('candidate_is_true requires boolean or 0/1 labels')
