"""Small real fits of the reranker trainers, plus input-isolation checks."""
import csv
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class RerankTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.logs = []
        spectra = cls.root / 'spec_files'
        spectra.mkdir()
        rows, roles = [], []
        for index in range(64):
            sid = f'train{index}'
            roles.append({'name': sid, 'split': 'train'})
            spectra.joinpath(f'{sid}.ms').write_text('>parentmass 200\n>ms2peaks\n172.00508538 100\n', encoding='utf-8')
            length, end = index // 2 + 1, ('O' if index % 2 else 'N')
            positive = 'C' * length + end
            for label, smiles in [(1, positive), (0, 'C' * length + ('F' if index % 2 else 'Cl'))]:
                rows.append({'spec_id': sid, 'candidate_hash': f'{index}_{label}',
                             'candidate_rank_freq': 2 - label, 'candidate_freq': 1 + label * 5,
                             'candidate_canonical_smiles': smiles,
                             'candidate_canonical_smiles_nostereo': smiles,
                             'true_canonical_smiles': positive, 'candidate_is_true': label,
                             'mass_frag_v2_candidate_common_loss_support': label,
                             'mass_frag_v2_common_loss_hits': 1,
                             'mass_frag_v2_candidate_common_loss_observed': label})
        table(cls.root / 'train.tsv', rows)
        valid = []
        for index in range(12):
            sid = f'val{index}'
            roles.append({'name': sid, 'split': 'val'})
            for original in rows[index * 2:index * 2 + 2]:
                valid.append({**original, 'spec_id': sid})
        table(cls.root / 'valid.tsv', valid)
        roles.append({'name': 'heldout', 'split': 'test'})
        table(cls.root / 'split.tsv', roles)
        table(cls.root / 'test.tsv', [{**rows[0], 'spec_id': 'heldout'}])
        table(cls.root / 'protected.tsv', [{'smiles': rows[0]['candidate_canonical_smiles']}])

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('RERANK_TEST_LOG'):
            Path(os.environ['RERANK_TEST_LOG']).write_text('\n'.join(cls.logs), encoding='utf-8')
        cls.temp.cleanup()

    def run_cli(self, module, *args, success=True):
        result = subprocess.run([sys.executable, '-m', 'rerank.' + module, *map(str, args)],
                                cwd=REPO, capture_output=True, text=True)
        self.logs.append(module + '\n' + result.stdout + result.stderr)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_01_rankers_and_inference_contract(self):
        r = self.root
        self.run_cli('train_evidencerank', '--train-table', r/'train.tsv', '--valid-table', r/'valid.tsv',
                     '--split-table', r/'split.tsv', '--out-dir', r/'exact', '--features', 'candidate_freq',
                     '--n-estimators', 3, '--min-child-samples', 1, '--n-jobs', 1)
        for role in ('train', 'valid'):
            self.run_cli('build_similarity_targets', '--input-table', r/f'{role}.tsv',
                         '--output-targets', r/f'{role}_targets.tsv', '--summary-json', r/f'{role}_targets.json',
                         '--split-table', r/'split.tsv', '--declared-split', 'train' if role == 'train' else 'val')
        self.run_cli('train_stage_a', '--train-table', r/'train.tsv', '--valid-table', r/'valid.tsv',
                     '--split-table', r/'split.tsv', '--out-dir', r/'stage_a',
                     '--train-targets', r/'train_targets.tsv', '--valid-targets', r/'valid_targets.tsv',
                     '--features-from-metrics', r/'exact/metrics.json', '--exact-valid-scores', r/'exact/valid_scores.tsv',
                     '--n-estimators', 3, '--min-child-samples', 1, '--n-jobs', 1)
        with (r/'stage_a/stage_a_model.pkl').open('rb') as handle:
            stage = pickle.load(handle)
        self.run_cli('score', 'stage-a', '--candidate-table', r/'valid.tsv',
                     '--model-pkl', r/'stage_a/stage_a_model.pkl', '--evidencerank-scores', r/'exact/valid_scores.tsv',
                     '--out-scores', r/'inference.tsv', '--declared-split', 'test',
                     '--exact-weight', stage['selected_exact_weight'])
        import pandas as pd
        inferred = pd.read_csv(r/'inference.tsv', sep='\t').score
        fitted = pd.read_csv(r/'stage_a/valid_selected_fused_scores.tsv', sep='\t').score
        np.testing.assert_allclose(inferred, fitted, atol=1e-6)
        self.assertTrue((r/'exact/training_inputs.json').is_file())

    def test_02_auxiliaries_and_windows_pool(self):
        r = self.root
        for name, out in [('train_support_surrogate', 'support'), ('train_loss_support_flags', 'flags')]:
            self.run_cli(name, '--train-table', r/'train.tsv', '--split-table', r/'split.tsv',
                         '--out-dir', r/out, '--data-root', r, '--workers', 2, '--n-jobs', 1,
                         '--n-estimators', 3, '--exclude-structures', r/'protected.tsv')
        from rerank.training_auxiliary import fingerprint
        x = np.stack([fingerprint('CCO'), fingerprint('CCN')])
        with (r/'support/massfrag_support_surrogate.pkl').open('rb') as handle:
            bundle = pickle.load(handle)
        self.assertEqual(bundle['model'].predict(x).shape, (2,))
        with (r/'flags/massfrag_loss_flag_models.pkl').open('rb') as handle:
            flags = pickle.load(handle)
        self.assertIn('CO', flags['models'])
        self.assertEqual(flags['models']['CO'].predict_proba(x).shape, (2, 2))
        ledger = (r/'support/structure_split.tsv').read_text()
        self.assertIn('internal_holdout', ledger)
        self.assertEqual(json.loads((r/'support/training_inputs.json').read_text())['excluded_rows'], 1)

    def test_03_test_spectra_rejected(self):
        result = self.run_cli('train_support_surrogate', '--train-table', self.root/'test.tsv',
                             '--split-table', self.root/'split.tsv', '--out-dir', self.root/'bad', success=False)
        self.assertIn('is not train', result.stderr)
        self.assertFalse((self.root/'bad').exists())

    def test_04_features_and_protected_candidates(self):
        from rerank import training_common as tc
        with self.assertRaises(ValueError):
            tc.validate_features([{'true_score': 1, 'candidate_is_true': 1}], ['true_score'])
        with self.assertRaises(ValueError):
            tc.validate_features([{'candidate_is_true': 1}], ['candidate_freq'])
        protected = tc.protected_structures([self.root/'protected.tsv'])
        with self.assertRaisesRegex(ValueError, 'protected structure'):
            tc.validate_table_role(self.root/'train.tsv', tc.split_roles(self.root/'split.tsv'), 'train', protected)


if __name__ == '__main__':
    unittest.main()
