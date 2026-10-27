"""Shared decoder dataset selection and launcher regression checks."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from utils.config import load_yaml
from encoder.config import load as load_encoder_config, path as encoder_path


REPO = Path(__file__).resolve().parents[1]
STAGES = ("graph2mol", "ms2mol", "sample")


class SharedConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.assets = Path(self.tmp.name).resolve()
        self.env = patch.dict(os.environ, {"MS2FORGE_ASSETS": str(self.assets),
                                           "DATASET": "nplib1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def load(self, stage, dataset=None, edit=False):
        text = (REPO / "configs" / (stage + ".yml")).read_text(encoding="utf-8")
        if edit:
            text = text.replace("dataset: msg", "dataset: nplib1", 1)
        return load_yaml(io.StringIO(text), dataset=dataset)

    def test_yaml_default_overrides_unrelated_environment_dataset(self):
        for stage in STAGES:
            cfg = self.load(stage)
            self.assertEqual(cfg["dataset"], "msg")
            expected = (Path(self.load("rerank")["candidate_globs"]["test"]).parent
                        if stage == "sample" else self.assets / "msg" / "output" / stage)
            self.assertEqual(Path(cfg["output"]), expected)
            self.assertEqual(Path(cfg['encoder_checkpoint']), self.assets/'msg/embeddings/encoder.json')

    def test_manual_switch_and_explicit_override_select_all_dataset_paths(self):
        for stage in STAGES:
            cfg = self.load(stage, edit=True)
            self.assertEqual(cfg, self.load(stage, dataset="nplib1"))
            expected = (Path(self.load("rerank", "nplib1")["candidate_globs"]["test"]).parent
                        if stage == "sample" else self.assets / "nplib1" / "output" / stage)
            self.assertEqual(Path(cfg["output"]), expected)
            self.assertEqual(Path(cfg['encoder_checkpoint']), self.assets/'nplib1/embeddings/encoder.json')
            if stage != "sample":
                self.assertEqual(Path(cfg["data"]["train"]),
                                 self.assets / "nplib1" / "packed" / stage / "train")
            else:
                self.assertEqual(Path(cfg["checkpoint"]),
                                 self.assets / "nplib1" / "output" / "ms2mol" / "latest.json")
            self.assertEqual(self.load(stage, dataset="msg", edit=True), self.load(stage))

    def test_shared_parent_is_independent_of_dataset(self):
        msg, npl = self.load("ms2mol"), self.load("ms2mol", "nplib1")
        self.assertEqual(msg["parent"], npl["parent"])
        self.assertEqual(Path(msg["parent"]),
                         self.assets / 'shared/checkpoints/graph2mol.json')

    def test_encoder_pretraining_and_finetuning_share_weight_path(self):
        cfg = load_encoder_config(REPO / 'configs/encoder.yml')
        expected = self.assets / 'shared/checkpoints/encoder/clean_encoder.pth'
        self.assertEqual(encoder_path(cfg['pretrain']['weights']), expected)
        self.assertEqual(encoder_path(cfg['finetune']['initial_weights']), expected)

    def test_invalid_dataset_and_unresolved_variables_are_rejected(self):
        with self.assertRaises(ValueError):
            self.load("sample", dataset="invalid")
        with self.assertRaises(ValueError):
            load_yaml(io.StringIO("dataset: msg\noutput: ${MISSING_CONFIG_TEST_VAR_123}\n"))

    def test_launcher_uses_shared_configs_with_optional_dataset(self):
        for stage in STAGES:
            for dataset in (None, "msg", "nplib1"):
                args = [] if dataset is None else [dataset]
                result = subprocess.run(
                    [sys.executable, str(REPO / "run.py"), *args, stage, "--dry-run"],
                    cwd=REPO, capture_output=True, text=True, check=True)
                self.assertIn("configs/" + stage + ".yml", result.stdout)
                if dataset is None:
                    self.assertNotIn("--dataset", result.stdout)
                else:
                    self.assertIn("--dataset " + dataset, result.stdout)


if __name__ == "__main__":
    unittest.main()
