import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rerank.build_candidates import BFN_COLUMNS
from rerank.build_pairs import aligned_groups
from rerank.mass_features import mass_row
from rerank.prepare_labels import audit_splits
from rerank.spectra import read_spectrum
from rerank.training_common import canonical, sha256

REPO = Path(__file__).resolve().parents[1]


def write_table(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class RerankWorkflowTests(unittest.TestCase):
    def cli(self, module, *args, env=None):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=REPO,
                                env=env, capture_output=True, text=True, timeout=240)
        if result.returncode:
            self.fail(result.stdout + result.stderr)
        return result

    def fixture(self, root, dataset):
        data = root / dataset / 'data'
        (data / 'spec_files').mkdir(parents=True)
        labels, splits = [], []
        molecules = {'train': ['CCO', 'CCN', 'CCF', 'CCC', 'CCCl', 'CCBr'],
                     'val': ['CCCO', 'CCCN'], 'test': ['CCCF', 'CCCCCl']}
        for role, smiles in molecules.items():
            folder = root / dataset / 'output/rerank_candidates' / role
            folder.mkdir(parents=True)
            with (folder / 'candidates_rank0.jsonl').open('w', encoding='utf-8') as f:
                for i, truth in enumerate(smiles):
                    sid = f'{role}{i}'
                    mol = Chem.MolFromSmiles(truth)
                    parent = rdMolDescriptors.CalcExactMolWt(mol) + 1.007276466621
                    labels.append(dict(spec=sid, smiles=truth, formula=rdMolDescriptors.CalcMolFormula(mol),
                                       ionization='[M+H]+', instrument='Orbitrap'))
                    splits.append(dict(name=sid, split=role))
                    header = '>ms2peaks' if dataset == 'msg' else '>collision 20'
                    (data / 'spec_files' / f'{sid}.ms').write_text(
                        f'>parentmass {parent}\n>ionization [M + H]+\n>ms1peaks\n999 999\n{header}\n{parent-18.010565} 100\n20 30\n', encoding='utf-8')
                    if role == 'test' and i == 1:
                        f.write(json.dumps(dict(type='input_failure', spec_id=sid, reason='fixture')) + '\n')
                        continue
                    candidates = [canonical(truth), 'CO', 'CN']
                    if role == 'train' and i == 5:
                        candidates = ['CO', 'CN']
                    counts = [[s, 3-j] for j, s in enumerate(candidates)]
                    stats = [dict(candidate_canonical_smiles_nonisomeric=s,
                                  **{name: float(3-j) for name in BFN_COLUMNS}) for j, s in enumerate(candidates)]
                    record = dict(type='mol_candidates', schema='ms2forge_candidate_cache.v1', spec_id=sid,
                                  pred=dict(n_generated=6, connected_smiles_nonisomeric_counts=counts,
                                            connected_smiles_counts=counts, bfn_stats_nonisomeric=stats,
                                            candidate_first_seen_nonisomeric=[[s, j+1] for j, s in enumerate(candidates)]))
                    f.write(json.dumps(record) + '\n')
        write_table(data / 'labels.tsv', labels)
        write_table(data / 'split.tsv', splits)
        return data

    def test_full_pipeline_dataset_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = yaml.safe_load((REPO / 'configs/rerank.yml').read_text(encoding='utf-8'))
            cfg['workers'] = 1
            cfg['evidencerank'] = dict(n_estimators=3, min_child_samples=1)
            cfg['stage_a'] = dict(n_estimators=3, min_child_samples=1)
            cfg['ssr'] = dict(epochs=1, batch_size=4, workers=0, embedding_dim=8, monitor_fraction=.34)
            config = root / 'workflow.yml'
            config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            env = {**os.environ, 'MS2FORGE_ASSETS': str(root), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            for dataset in ('msg', 'nplib1'):
                with self.subTest(dataset=dataset):
                    data = self.fixture(root, dataset)
                    args = [] if dataset == 'msg' else [dataset]
                    self.cli('rerank.train', *args, '--config', config, env=env)
                    out = root / dataset / 'output/rerank_training'
                    if dataset == 'msg':
                        repeated = root / 'repeat_training'
                        self.cli('rerank.train', '--config', config, '--output-dir', repeated, env=env)
                        import torch
                        left = torch.load(out / 'ssr_model/best.pt', map_location='cpu', weights_only=False)
                        right = torch.load(repeated / 'ssr_model/best.pt', map_location='cpu', weights_only=False)
                        for key in left['state_dict']:
                            self.assertTrue(torch.equal(left['state_dict'][key], right['state_dict'][key]), key)
                        a = pd.read_csv(out / 'selection/valid_selected_scores.tsv', sep='\t')
                        b = pd.read_csv(repeated / 'selection/valid_selected_scores.tsv', sep='\t')
                        np.testing.assert_array_equal(a.score, b.score)
                    manifest = json.loads((out / 'models/manifest.json').read_text())
                    self.assertEqual(manifest['dataset'], dataset)
                    self.assertFalse(manifest['test_used_for_training'])
                    id_audit = json.loads((out / 'input_id_audit.json').read_text())
                    self.assertEqual(manifest['input_id_audit'], id_audit)
                    self.assertEqual(id_audit['test_id_count'], 2)
                    self.assertEqual(id_audit['train_input_id_count'], 6)
                    self.assertEqual(id_audit['validation_input_id_count'], 2)
                    self.assertEqual(id_audit['train_test_overlap_count'], 0)
                    self.assertEqual(id_audit['validation_test_overlap_count'], 0)
                    split_rows = pd.read_csv(data / 'split.tsv', sep='\t')
                    test_ids = set(split_rows.loc[split_rows.split.eq('test'), 'name'])
                    for path, entry in id_audit['tables'].items():
                        actual_ids = set(pd.read_csv(path, sep='\t').spec_id)
                        self.assertEqual(entry['unique_ids'], len(actual_ids))
                        self.assertEqual(entry['test_overlap_count'], len(actual_ids & test_ids))
                        self.assertEqual(entry['sha256'], sha256(path))
                    metrics = json.loads((out / 'exact/metrics.json').read_text())
                    self.assertNotIn('test_table', metrics)
                    self.assertNotIn('test_metrics', metrics)
                    self.assertNotIn('test_table', metrics['args'])
                    features = pd.read_csv(out / 'train/features.tsv', sep='\t')
                    self.assertNotIn('candidate_is_true', features)
                    self.assertEqual(json.loads((out / 'train/features.json').read_text())['n_features'], 241)
                    pairs = pd.read_csv(out / 'train/pairs.tsv', sep='\t', keep_default_na=False)
                    self.assertFalse(pairs.chosen_candidate_hash.eq('').any())
                    self.assertTrue(pairs.truth_candidate_present.eq(0).any())
                    self.assertEqual(len(set(np.load(out / 'ssr_train/chosen_group.npy'))), 6)
                    self.assertFalse((out / 'models/massfrag_support_surrogate.pkl').exists())
                    models = out / 'models'
                    for repeat in range(2):
                        if repeat:
                            refs = pd.read_csv(data / 'labels.tsv', sep='\t')
                            refs.loc[refs.spec.str.startswith('test'), 'smiles'] = 'c1ccccc1'
                            refs.to_csv(data / 'labels.tsv', sep='\t', index=False)
                        self.cli('rerank.run', dataset, '--cache-glob', root / dataset / 'output/rerank_candidates/test/*.jsonl',
                                 '--model-dir', models, '--data-root', data, '--output-dir', root / f'{dataset}_test{repeat}',
                                 '--workers', 1, env=env)
                    first = pd.read_csv(root / f'{dataset}_test0/scores.tsv', sep='\t')
                    second = pd.read_csv(root / f'{dataset}_test1/scores.tsv', sep='\t')
                    np.testing.assert_array_equal(first.score, second.score)
                    scores_meta = json.loads((root / f'{dataset}_test0/scores.json').read_text())
                    self.assertAlmostEqual(scores_meta['fusion']['stage_a_weight'], manifest['ssr_base_weight'])
                    status = pd.read_csv(root / f'{dataset}_test0/spectrum_status.tsv', sep='\t')
                    self.assertEqual(len(status), 2)

    def test_sampling_preparation_and_wrapper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = {**os.environ, 'MS2FORGE_ASSETS': str(root)}
            for dataset in ('msg', 'nplib1'):
                self.fixture(root, dataset)
                args = [] if dataset == 'msg' else [dataset]
                self.cli('rerank.prepare_sampling', *args, env=env)
                for role in ('train', 'val', 'test'):
                    folder = root / dataset / 'output/rerank_candidates' / role
                    current = yaml.safe_load((folder / 'sample.yml').read_text())
                    self.assertEqual(current['dataset'], dataset)
                    self.assertIn(dataset, current['checkpoint'])
                    inputs = pd.read_csv(folder / 'generation_inputs.tsv', sep='\t')
                    self.assertTrue(inputs.split.eq(role).all())
                    self.assertNotIn('smiles', inputs)
                result = subprocess.run([sys.executable, str(REPO / 'rerank.py'), *args, '--dry-run'],
                                        cwd=REPO, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('rerank_training', result.stdout)
                self.assertIn('rerank_candidates', result.stdout)

    def test_spectra_and_direct_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'spec_files'
            path.mkdir()
            spectrum = path / 'sample.ms'
            spectrum.write_text('>parentmass 100\n>ionization [M + H]+\n>ms1peaks\n999 9\n>collision 10\n72.005085 100\n>collision 20\n72.005085 50\n', encoding='utf-8')
            parent, ion, peaks = read_spectrum(spectrum)
            self.assertEqual(peaks, ((72.005085, 150.0),))
            self.assertEqual(ion, '[M+H]+')
            row = mass_row(('sample', 'CC=O', temp))
            self.assertEqual(row['mass_frag_v2_candidate_common_loss_support'], sum(row[f'mass_frag_v2_candidate_loss_{x}_support'] for x in ('H2O','NH3','CO','CO2','CH2O','C2H4','SO2')))
            spectrum.write_text('>parentmass 100\n>ms1peaks\n50 10\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_spectrum(spectrum)

    def test_reject_leakage_and_truncated_pair_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_table(root / 'labels.tsv', [dict(spec='a', smiles='CCO'), dict(spec='b', smiles='OCC')])
            write_table(root / 'split.tsv', [dict(name='a', split='train'), dict(name='b', split='val')])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                audit_splits(root / 'labels.tsv', root / 'split.tsv')
            row = dict(spec_id='a', candidate_hash='x', candidate_tanimoto_to_truth=1, score=1)
            for name, rows in [('a', [row, row]), ('b', [row]), ('c', [row])]:
                write_table(root / name, rows)
            with self.assertRaisesRegex(ValueError, 'row counts'):
                list(aligned_groups(root / 'a', root / 'b', root / 'c'))


if __name__ == '__main__':
    unittest.main()
