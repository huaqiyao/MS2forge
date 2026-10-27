"""Build train-only SSR pairs from aligned candidates, targets and ER scores."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from rerank import training_common as tc
from rerank.prepare_labels import load_references, canonical


def aligned_groups(table, targets, scores):
    with Path(table).open(encoding='utf-8', newline='') as a, Path(targets).open(encoding='utf-8', newline='') as b, Path(scores).open(encoding='utf-8', newline='') as c:
        readers = [csv.DictReader(f, delimiter='\t') for f in (a, b, c)]
        seen, current, rows = set(), None, []
        for triple in itertools.zip_longest(*readers):
            if any(r is None for r in triple):
                raise ValueError('Candidate/target/score row counts differ')
            row, target, score = triple
            keys = [(r['spec_id'], r['candidate_hash']) for r in triple]
            if len(set(keys)) != 1:
                raise ValueError(f'Candidate/target/score key mismatch: {keys}')
            sid = row['spec_id']
            if current != sid:
                if current is not None:
                    yield current, rows
                if sid in seen:
                    raise ValueError(f'Noncontiguous spectrum group: {sid}')
                seen.add(sid)
                current, rows = sid, []
            row['_score'] = float(score['score'])
            row['_tan'] = float(target['candidate_tanimoto_to_truth'])
            if not math.isfinite(row['_score']) or not 0 <= row['_tan'] <= 1:
                raise ValueError('Invalid score or Tanimoto')
            rows.append(row)
        if current is not None:
            yield current, rows


def build_pairs(table, targets, scores, labels, split_table, output):
    roles, references = tc.split_roles(split_table), load_references(labels)
    tc.validate_table_role(table, roles, 'train')
    fields = ['spec_id', 'chosen_smiles', 'chosen_candidate_hash', 'rejected_smiles',
              'rejected_candidate_hash', 'rejected_tanimoto', 'preference_gap', 'truth_candidate_present']
    counts = Counter()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for sid, rows in aligned_groups(table, targets, scores):
            truth = canonical(references[sid])
            for row in rows:
                exact = canonical(row['candidate_canonical_smiles']) == truth
                if exact != (str(row['candidate_is_true']).lower() in {'1', 'true', 'yes'}):
                    raise ValueError(f'Incorrect candidate label for {sid}')
            wrong = [r for r in rows if canonical(r['candidate_canonical_smiles']) != truth]
            if not wrong:
                counts['no_negative'] += 1
                continue
            rejected = min(wrong, key=lambda r: (-r['_score'], float(r['candidate_rank_freq']), r['candidate_hash']))
            present = len(wrong) < len(rows)
            writer.writerow(dict(spec_id=sid, chosen_smiles=truth,
                                 chosen_candidate_hash=hashlib.sha256(truth.encode()).hexdigest(),
                                 rejected_smiles=rejected['candidate_canonical_smiles'],
                                 rejected_candidate_hash=rejected['candidate_hash'],
                                 rejected_tanimoto=rejected['_tan'], preference_gap=1-rejected['_tan'],
                                 truth_candidate_present=int(present)))
            counts['pairs'] += 1
            counts['truth_absent'] += not present
    if not counts['pairs']:
        raise ValueError('No SSR pairs: train candidates need incorrect alternatives')
    summary = {'source_split': 'train', 'counts': dict(counts),
               'input_sha256': {str(p): tc.sha256(p) for p in (table, targets, scores, labels, split_table)}}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-table', 'targets', 'scores', 'labels', 'split-table', 'out-tsv'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    build_pairs(a.candidate_table, a.targets, a.scores, a.labels, a.split_table, a.out_tsv)


if __name__ == '__main__':
    main()
