"""Behavioral checks for held-out spectrum isolation and the training CLI."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rerank.training_common import audit_input_ids, sha256

REPO = Path(__file__).resolve().parents[1]


def write_table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class InputIdAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / 'split.tsv'
        self.train = self.root / 'train.tsv'
        self.valid = self.root / 'valid.tsv'
        write_table(self.split, [dict(name='train0', split='train'), dict(name='unused', split='train'),
                                dict(name='val0', split='val'), dict(name='test0', split='test')])
        write_table(self.train, [dict(spec_id='train0'), dict(spec_id='train0')])
        write_table(self.valid, [dict(spec_id='val0')])

    def test_counts_and_hashes_come_from_input_tables(self):
        result = audit_input_ids(self.split, [self.train], [self.valid])
        self.assertEqual(result['train_input_id_count'], 1)
        self.assertEqual(result['validation_input_id_count'], 1)
        self.assertEqual(result['test_id_count'], 1)
        self.assertEqual(result['train_test_overlap_count'], 0)
        self.assertEqual(result['validation_test_overlap_count'], 0)
        self.assertFalse(result['test_used_for_training'])
        self.assertFalse(result['test_used_for_selection'])
        self.assertEqual(result['split_sha256'], sha256(self.split))
        entry = result['tables'][str(self.train.resolve())]
        self.assertEqual(entry['rows'], 2)
        self.assertEqual(entry['sha256'], sha256(self.train))
        # A second genuine training input changes the audit, not the split file.
        pairs = self.root / 'pairs.tsv'
        write_table(pairs, [dict(spec_id='unused')])
        updated = audit_input_ids(self.split, [self.train, pairs], [self.valid])
        self.assertEqual(updated['train_input_id_count'], 2)
        json.dumps(updated)

    def test_test_ids_rejected_in_training_validation_and_pairs(self):
        for name in ('train.tsv', 'valid.tsv', 'pairs.tsv'):
            with self.subTest(table=name):
                path = self.root / name
                original = path.read_bytes() if path.exists() else None
                write_table(path, [dict(spec_id='test0')])
                train_tables = [self.train] + ([path] if name == 'pairs.tsv' else [])
                with self.assertRaisesRegex(ValueError, 'test IDs'):
                    audit_input_ids(self.split, train_tables, [self.valid])
                if original is not None:
                    path.write_bytes(original)

    def test_unknown_and_wrong_role_ids_rejected(self):
        for sid in ('unknown', 'val0', ''):
            with self.subTest(spec_id=sid):
                write_table(self.train, [dict(spec_id=sid)])
                with self.assertRaisesRegex(ValueError, 'outside declared train'):
                    audit_input_ids(self.split, [self.train], [self.valid])

    def test_absent_test_split_cannot_certify_isolation(self):
        write_table(self.split, [dict(name='train0', split='train'), dict(name='val0', split='val')])
        with self.assertRaisesRegex(ValueError, 'nonempty test split'):
            audit_input_ids(self.split, [self.train], [self.valid])

    def test_empty_or_malformed_tables_rejected(self):
        for content in ('spec_id\n', 'wrong_column\ntrain0\n'):
            with self.subTest(content=content):
                self.train.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit_input_ids(self.split, [self.train], [self.valid])

    def test_training_cli_rejects_removed_test_argument_before_fitting(self):
        out = self.root / 'model'
        result = subprocess.run(
            [sys.executable, '-m', 'rerank.train_evidencerank',
             '--train-table', str(self.train), '--valid-table', str(self.valid),
             '--split-table', str(self.split), '--out-dir', str(out),
             '--test-table', str(self.root / 'does_not_exist.tsv')],
            cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments: --test-table', result.stderr)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
