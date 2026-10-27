"""Small executable checks for exports, train/val separation and exact resume.

Run: python -m unittest discover -s tests -p test_encoder.py -v
Optional CUDA smoke tests: ENCODER_TEST_GPU_IDS=0,1 (visible device indices).
"""
import copy
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from encoder.config import load
from encoder.data import train_plan, monitor_indices
from encoder.memory import enable_activation_checkpointing
from encoder.model import build, TensorForward, finetune_loss, load_weights
from encoder.state import export, model_hash, seed_all, sha256, save
from encoder.train import run_pretrain, run_finetune, parallel_forward


class EncoderTests(unittest.TestCase):
    def test_all_stage_selects_new_pretraining_weights(self):
        from encoder.train import main
        cfg = self.config()
        cfg['stage'] = 'all'
        cfg['pretrain']['weights'] = str(self.root / 'new_pretrained.pth')
        cfg['finetune']['initial_weights'] = str(self.root / 'old_pretrained.pth')
        seen = []
        with mock.patch.object(sys, 'argv', ['encoder.train', '--stage', 'all']), \
             mock.patch('encoder.train.load', return_value=cfg), \
             mock.patch('encoder.train.setup_runtime', return_value=(torch.device('cpu'), [])), \
             mock.patch('encoder.train.run_pretrain', side_effect=lambda c,*a: seen.append(c['finetune']['initial_weights'])), \
             mock.patch('encoder.train.run_finetune', side_effect=lambda c,*a: seen.append(c['finetune']['initial_weights'])):
            main()
        self.assertEqual(seen, [str(self.root/'old_pretrained.pth'), str(self.root/'new_pretrained.pth')])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='encoder_test_')
        self.root = Path(self.temp.name)
        gpu_text = os.environ.get('ENCODER_TEST_GPU_IDS')
        self.gpus = [int(value) for value in gpu_text.split(',')] if gpu_text else []
        self.device = torch.device(f'cuda:{self.gpus[0]}' if self.gpus else 'cpu')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    def tearDown(self):
        self.temp.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def config(self):
        cfg = load('configs/encoder.yml')
        cfg['runtime'] = {'device': self.device.type, 'gpu_ids': self.gpus or [0], 'threads': 2}
        cfg['output_root'] = str(self.root / 'outputs')
        return cfg

    def test_initialization_and_pure_export(self):
        seed_all(42, 2)
        model = build(15)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 13550521)
        # Initial random tensors may differ across PyTorch versions/platforms.
        # Check reproducibility in this runtime instead of a platform-specific hash.
        initial_hash = model_hash(model)
        seed_all(42, 2)
        self.assertEqual(model_hash(build(15)), initial_hash)
        seed_all(43, 2)
        self.assertNotEqual(model_hash(build(15)), initial_hash)
        target = self.root / 'model.pth'
        export(target, model)
        state = torch.load(target, map_location='cpu', weights_only=True)
        self.assertEqual(len(state), 288)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in state.values()))
        probe = build()
        load_weights(probe, target)
        self.assertEqual(model_hash(probe), model_hash(model))

    def test_plan_and_monitor(self):
        train = np.arange(17)
        val = np.arange(20, 60)
        a = train_plan(train, 19, 8, 2026)
        b = train_plan(train, 29, 8, 2026)
        np.testing.assert_array_equal(a, b[:19])
        self.assertTrue(set(a[a >= 0]).issubset(set(train)))
        monitor = monitor_indices(val, .1, 2026)
        self.assertEqual(len(monitor), 4)
        self.assertTrue(set(monitor).issubset(set(val)))

    def batch(self, batch_size=8):
        device = self.device
        sos = torch.zeros(batch_size, 1, 13, device=device)
        sos[:, :, 0] = sos[:, :, 2] = 1
        formula = torch.randn(batch_size, 128, 144, device=device)
        mask = torch.ones(batch_size, 129, dtype=torch.bool, device=device)
        mask[:, :4] = False
        X = torch.zeros(batch_size, 2, 11, device=device)
        X[:, :, 1] = 1
        E = torch.zeros(batch_size, 2, 2, 5, device=device)
        E[:, :, :, 0] = 1
        return sos, formula, mask, X, E, torch.ones(batch_size, 1, device=device), torch.ones(batch_size, 2, dtype=torch.bool, device=device)

    def test_parallel_forward_and_checkpointing(self):
        seed_all(42, 2)
        model = build().to(self.device)
        inputs = self.batch()
        model.eval()
        parallel = parallel_forward(model, self.gpus)
        with torch.no_grad():
            expected = TensorForward(model)(*inputs)
            actual = parallel(*inputs)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        model.train()
        enable_activation_checkpointing(model)
        loss = finetune_loss(model, parallel(*inputs))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for prefix in ('ms_encoder', 'graph_encoder'):
            grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def finetune_fixture(self, cfg):
        data = self.root / 'features'
        data.mkdir()
        rng = np.random.RandomState(7)
        arrays = {'counts': rng.randint(0, 4, (32, 128, 9)).astype(np.int16),
                  'nodes': np.ones((4, 2), dtype=np.int8),
                  'edges': np.zeros((4, 2, 2), dtype=np.uint8),
                  'node_counts': np.full(4, 2, dtype=np.int16),
                  'graph_ids': np.arange(32, dtype=np.int64) % 4,
                  'adduct_ids': np.arange(32, dtype=np.int64) % 2,
                  'peak_counts': np.full(32, 3, dtype=np.int16),
                  'energy_bins': np.arange(32, dtype=np.int64) % 11,
                  'train_indices': np.arange(16, dtype=np.int64),
                  'val_indices': np.arange(16, 32, dtype=np.int64)}
        for name, value in arrays.items():
            np.save(data / f'{name}.npy', value)
        initial = self.root / 'initial.pth'
        seed_all(42, 2)
        export(initial, build())
        cfg['finetune'].update(initial_weights=str(initial), batch_size=4, warmup_steps=2,
                               monitor_every=2, monitor_ratio=.5)
        cfg['datasets']['msg'].update(prepared_dir=str(data), steps=4, weights_dir=str(self.root / 'weights_a'))
        return cfg

    def test_finetune_resume_matches_continuous(self):
        cfg = self.finetune_fixture(self.config())
        run_finetune(cfg, self.device, self.gpus)
        uninterrupted = torch.load(self.root / 'weights_a/final_step4.pth', weights_only=True, map_location='cpu')
        other = copy.deepcopy(cfg)
        other['output_root'] = str(self.root / 'outputs_b')
        other['datasets']['msg']['weights_dir'] = str(self.root / 'weights_b')
        other['stage'] = 'all'
        run_finetune(other, self.device, self.gpus, stop_after=2)
        other['stage'] = 'finetune'
        run_finetune(other, self.device, self.gpus, resume=True)
        resumed = torch.load(self.root / 'weights_b/final_step4.pth', weights_only=True, map_location='cpu')
        for name in uninterrupted:
            self.assertTrue(torch.equal(uninterrupted[name], resumed[name]), name)
        self.assertTrue((self.root / 'outputs_b/msg_finetune/resume.pth').exists())
        cfg['dataset'] = 'nplib1'
        cfg['datasets']['nplib1'].update(prepared_dir=cfg['datasets']['msg']['prepared_dir'], steps=3,
                                        weights_dir=str(self.root / 'weights_npl'))
        run_finetune(cfg, self.device, self.gpus)
        self.assertTrue((self.root / 'weights_npl/iter00002.pth').exists())
        with self.assertRaises(FileExistsError):
            run_finetune(cfg, self.device, self.gpus)

    def test_pretrain_and_export(self):
        cfg = self.config()
        rows, graphs = [], {}
        for i in range(40):
            smiles = f'fixture_{i:02d}'
            rows.append({'smiles': smiles, 'molecular_formula': 'C2H6', 'precursor_type': '[M+H]+',
                         'collision_energy_NCE': 20,
                         'clean_spectrum_formula_array': [[1, 1, 0, 0, 0, 0, 0, 0, 0]]})
            graph = Data(x=torch.nn.functional.one_hot(torch.tensor([1, 1]), 11).float(),
                         edge_index=torch.tensor([[0, 1], [1, 0]]),
                         edge_attr=torch.nn.functional.one_hot(torch.tensor([1, 1]), 5).float())
            graphs[smiles] = [np.array([i]), graph, torch.zeros(1, 2048)]
        parquet = self.root / 'fraghub.parquet'
        pd.DataFrame(rows).to_parquet(parquet)
        graph_file = self.root / 'graphs.pt'
        torch.save(graphs, graph_file)
        cfg['pretrain'].update(parquet=str(parquet), parquet_sha256=sha256(parquet),
                               original_parquet=str(parquet), original_graphs=str(graph_file),
                               prepared_dir=str(self.root / 'prepared'), epochs=2, batch_size=8,
                               ordered_sub_batch_size=2, warmup_steps=2, save_every_epochs=1,
                               weights=str(self.root / 'pretrained.pth'))
        run_pretrain(cfg, self.device, self.gpus)
        exported = torch.load(self.root / 'pretrained.pth', weights_only=True, map_location='cpu')
        self.assertEqual(len(exported), 288)
        self.assertTrue(all(torch.isfinite(value).all() for value in exported.values()))
        self.assertTrue((self.root / 'outputs/pretrain/resume.pth').exists())
        report = json.loads((self.root / 'outputs/pretrain/COMPLETED.json').read_text())
        self.assertEqual(report['updates'], 10)
        resumed_cfg = copy.deepcopy(cfg)
        resumed_cfg['output_root'] = str(self.root / 'resume_outputs')
        resumed_cfg['pretrain']['weights'] = str(self.root / 'resumed_pretrained.pth')

        def interrupt_after_epoch(filename, obj):
            save(filename, obj)
            if Path(filename).name == 'resume.pth' and obj['completed_epochs'] == 1:
                raise InterruptedError('Simulated stop after an atomic epoch checkpoint')

        with mock.patch('encoder.train.save', side_effect=interrupt_after_epoch):
            with self.assertRaises(InterruptedError):
                run_pretrain(resumed_cfg, self.device, self.gpus)
        run_pretrain(resumed_cfg, self.device, self.gpus, resume=True)
        resumed = torch.load(self.root / 'resumed_pretrained.pth', weights_only=True, map_location='cpu')
        for name in exported:
            self.assertTrue(torch.equal(exported[name], resumed[name]), name)


if __name__ == '__main__':
    unittest.main()
