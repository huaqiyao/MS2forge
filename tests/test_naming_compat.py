"""MS2Forge naming migration preserves configs, weights and candidate caches."""
import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml

from models import bfn, model
from rerank.build_candidates import build_candidates
from utils.compat import normalize_model_config
from utils.config import load_yaml
from utils.eval_utils import build_candidate_cache_record


REPO = Path(__file__).resolve().parents[1]


class NamingCompatibilityTests(unittest.TestCase):
    def test_config_alias_normalization_and_conflicts(self):
        legacy = {"flash": {"beta1": 7.25}, "stage": "ms2mol"}
        original = copy.deepcopy(legacy)
        current = {"ms2forge": {"beta1": 7.25}, "stage": "ms2mol"}
        self.assertEqual(normalize_model_config(legacy), current)
        self.assertEqual(legacy, original)
        ns = SimpleNamespace(flash=SimpleNamespace(beta1=7.25), stage="ms2mol")
        self.assertEqual(normalize_model_config(ns), current)
        for settings in (legacy, current, {**legacy, **current}):
            text = yaml.safe_dump({"dataset": "msg", "model": settings})
            self.assertEqual(load_yaml(io.StringIO(text))["model"], current)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_model_config({"flash": {"beta1": 3.0},
                                    "ms2forge": {"beta1": 7.25}})

    def test_legacy_import_and_pickle_class_paths(self):
        # Protocol-0 GLOBAL references match class paths in historical pickles.
        for module in (bfn, model):
            reference = ("c" + module.__name__ + "\nFLASH\n.").encode()
            self.assertIs(pickle.loads(reference), module.MS2Forge)
            self.assertIs(module.FLASH, module.MS2Forge)
            self.assertEqual(module.MS2Forge.__name__, "MS2Forge")

    def test_old_config_and_tensor_weights_load_without_changes(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for stage in ("graph2mol", "ms2mol"):
            with self.subTest(stage=stage):
                config = yaml.safe_load((REPO / 'configs' / (stage + '.yml')).read_text(encoding='utf-8'))
                config['model']['ms2forge']['beta1'] = 7.25
                legacy = copy.deepcopy(config)
                legacy['model']['flash'] = legacy['model'].pop('ms2forge')
                torch.manual_seed(42)
                old = model.build_eq_cond_model(legacy).eval()
                torch.manual_seed(42)
                new = model.build_eq_cond_model(config).eval()
                self.assertEqual(float(new.beta1), 7.25)
                self.assertEqual(tuple(old.state_dict()), tuple(new.state_dict()))
                for key, value in old.state_dict().items():
                    self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'existing.pt'
                    torch.save(old.state_dict(), path)
                    new.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
                del old, new

    def record(self):
        record = build_candidate_cache_record(
            [torch.tensor([1])], None, node_types=torch.tensor([1, 1]),
            halfedge_index=torch.tensor([[0], [1]]),
            pred_probs=[torch.tensor([[0.1, 0.6, 0.1, 0.1, 0.1]])])
        record.pop('true')
        record['spec_id'] = 's1'
        self.assertEqual(record['schema'], 'ms2forge_candidate_cache.v1')
        return record

    def test_old_new_and_mixed_candidate_cache_files(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = []
            for name, schema in [('old', 'flash_candidate_cache.v1'),
                                 ('new', 'ms2forge_candidate_cache.v1')]:
                folder = root / name
                folder.mkdir()
                cache = folder / 'candidates.jsonl'
                cache.write_text(json.dumps({**record, 'schema': schema}) + '\n', encoding='utf-8')
                original = cache.read_bytes()
                summary = build_candidates(cache, folder/'table.tsv', folder/'status.tsv', folder/'summary.json')
                self.assertEqual(summary['n_candidates'], 1)
                self.assertEqual(cache.read_bytes(), original)
                outputs.append(((folder/'table.tsv').read_bytes(), (folder/'status.tsv').read_bytes()))
            self.assertEqual(outputs[0], outputs[1])
            # Resumed sampling can append the new schema to an existing shard.
            mixed = root / 'mixed.jsonl'
            mixed.write_text(json.dumps({**record, 'schema': 'flash_candidate_cache.v1'}) + '\n' +
                             json.dumps({**record, 'spec_id': 's2'}) + '\n', encoding='utf-8')
            summary = build_candidates(mixed, root/'mixed.tsv', root/'status.tsv', root/'summary.json')
            self.assertEqual(summary['n_spectra'], 2)
            self.assertEqual(summary['n_candidates'], 2)
            bad = root / 'bad.jsonl'
            bad.write_text(json.dumps({**record, 'schema': 'unknown.v1'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unsupported cache schema'):
                build_candidates(bad, root/'bad.tsv', root/'bad_status.tsv', root/'bad_summary.json')


if __name__ == '__main__':
    unittest.main()
