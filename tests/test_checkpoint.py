"""Decoder weight format, lineage and optional exact-resume regressions."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from utils import checkpoint
from utils.artifacts import publish, require_encoder


class CheckpointTests(unittest.TestCase):
    def model(self, stage):
        model = nn.Module()
        model.formula_condition = nn.Linear(3, 2)
        model.register_buffer("beta1", torch.tensor(3.0))
        if stage == "ms2mol":
            model.zms_adapter = nn.Linear(2, 2)
        return model

    def test_default_dynamic_weights_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stage in ("graph2mol", "ms2mol"):
                with self.subTest(stage=stage):
                    model = self.model(stage)
                    # Newly registered parameters are included without a saved-key allowlist.
                    model.extra = nn.Linear(2, 1)
                    disabled = Mock()
                    disabled.state_dict.side_effect = AssertionError("Disabled state was collected")
                    path = Path(tmp) / (stage + ".pt")
                    cfg = {"stage": stage, "encoder_sha256": "encoder", "train": {}}
                    with patch.object(checkpoint, "rng_state", side_effect=AssertionError):
                        checkpoint.save(path, model, disabled, disabled, disabled, cfg, 12, 0, 1)
                    raw = torch.load(path, weights_only=True)
                    self.assertEqual(list(raw), list(model.state_dict()))
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, raw[name]))
                    self.model(stage).load_state_dict({k: v for k, v in raw.items() if not k.startswith("extra.")})
                    pointer = Path(tmp) / (stage + "_latest.json")
                    publish(pointer, path, stage, encoder_sha256="encoder")
                    for selected in (path, pointer):
                        payload, state = checkpoint.read_weights(selected)
                        require_encoder(payload, "encoder", "test")
                        self.assertEqual(list(state), list(raw))
                        with self.assertRaises(ValueError):
                            require_encoder(payload, "different", "test")
                    with self.assertRaisesRegex(ValueError, "--init"):
                        checkpoint.restore(path, model, disabled, disabled, disabled, cfg, 0, 1)
                    with self.assertRaises(FileExistsError):
                        checkpoint.save(path, model, disabled, disabled, disabled, cfg, 12, 0, 1)
                    sidecar = path.with_name(path.name + ".json")
                    meta = json.loads(sidecar.read_text())
                    meta["sha256"] = "wrong"
                    sidecar.write_text(json.dumps(meta))
                    with self.assertRaisesRegex(ValueError, "content changed"):
                        checkpoint.read_weights(path)

    def test_optional_states_restore_optimizer_scheduler_sampler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model("ms2mol")
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            sum(p.square().sum() for p in model.parameters()).backward()
            optimizer.step()
            scheduler.step(2.0)
            sampler = Mock()
            sampler.state_dict.return_value = {"visit": 7}
            fields = ["model", "optimizer", "scheduler", "sampler", "rng", "config", "update",
                      "schema", "architecture_id", "stage", "rank", "world", "encoder_sha256"]
            cfg = {"stage": "ms2mol", "encoder_sha256": "encoder",
                   "model": {"flash": {"beta1": 3.0}}, "train": {
                "checkpoint_fields": fields, "seed": 42, "global_batch": 2,
                "max_micro_graphs": 1, "max_micro_halfedges": 10, "warmup_updates": 1}}
            path = Path(tmp) / "full.pt"
            checkpoint.save(path, model, optimizer, scheduler, sampler, cfg, 7, 0, 1)
            # A naming-only config migration must preserve exact resume.
            cfg["model"] = {"ms2forge": {"beta1": 3.0}}
            expected_model = copy.deepcopy(model.state_dict())
            expected_optim = copy.deepcopy(optimizer.state_dict())
            expected_rng = (random.random(), np.random.rand(), torch.rand(3))
            fresh = self.model("ms2mol")
            fresh_optim = torch.optim.AdamW(fresh.parameters())
            fresh_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fresh_optim)
            fresh_sampler = Mock()
            update = checkpoint.restore(path, fresh, fresh_optim, fresh_scheduler, fresh_sampler, cfg, 0, 1)
            self.assertEqual(update, 7)
            fresh_sampler.load_state_dict.assert_called_once_with({"visit": 7})
            self.assertEqual(fresh_scheduler.state_dict(), scheduler.state_dict())
            for name, value in fresh.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[name]))
            restored_optim = fresh_optim.state_dict()
            self.assertEqual(restored_optim["param_groups"], expected_optim["param_groups"])
            for key, values in expected_optim["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(restored_optim["state"][key][name], value, rtol=0, atol=0)
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.rand(), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(3), expected_rng[2]))
            cfg["model"]["ms2forge"]["beta1"] = 4.0
            with self.assertRaisesRegex(ValueError, "Resume changes model"):
                checkpoint.restore(path, fresh, fresh_optim, fresh_scheduler,
                                   fresh_sampler, cfg, 0, 1)

    def test_selected_optional_fields_and_invalid_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model("graph2mol")
            cfg = {"stage": "graph2mol", "train": {"checkpoint_fields": ["model", "update"]}}
            path = Path(tmp) / "selected.pt"
            checkpoint.save(path, model, None, None, None, cfg, 5, 0, 1)
            self.assertEqual(set(torch.load(path, weights_only=True)), {"model", "update"})
            with self.assertRaisesRegex(ValueError, "--init"):
                checkpoint.restore(path, model, None, None, None, cfg, 0, 1)
            for fields in ([], ["optimizer"], ["model", "typo"], ["model", "model"], "model"):
                cfg["train"]["checkpoint_fields"] = fields
                with self.assertRaises(ValueError):
                    checkpoint.save(Path(tmp) / "bad.pt", model, None, None, None, cfg, 5, 0, 1)
            self.assertFalse((Path(tmp) / "bad.pt").exists())


if __name__ == "__main__":
    unittest.main()
