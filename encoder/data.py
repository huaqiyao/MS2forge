"""Prepared spectrum/graph tensors and reproducible train-only sampling."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .state import get_rng, set_rng
from .model import finetune_loss

ARRAY_NAMES = ('counts', 'nodes', 'edges', 'node_counts', 'graph_ids', 'adduct_ids', 'peak_counts', 'energy_bins')


class Inputs:
    def __init__(self, directory, device):
        self.directory = Path(directory)
        self.device = torch.device(device)
        if not (self.directory / 'energy_bins.npy').is_file():
            raise FileNotFoundError('Prepared encoder inputs lack energy_bins.npy; rerun encoder.prepare in a new prepared_dir')
        self.arr = {name: np.load(self.directory / f'{name}.npy', mmap_mode='r') for name in ARRAY_NAMES}
        energy = self.arr['energy_bins']
        if (energy.shape != (len(self.arr['counts']),) or
                not np.issubdtype(energy.dtype, np.integer) or
                (energy < 0).any() or (energy > 10).any()):
            raise ValueError('energy_bins.npy must contain one integer in [0, 10] per spectrum')
        self.gpu = {name: torch.tensor(np.array(value), device=self.device) for name, value in self.arr.items()}
        div = torch.exp(torch.arange(0, 16, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / 8))
        positions = torch.arange(150, dtype=torch.float32)[:, None]
        lut = torch.zeros(150, 16)
        lut[:, 0::2] = torch.sin(positions * div)
        lut[:, 1::2] = torch.cos(positions * div)
        self.lut = lut.to(self.device)

    def indices(self, split):
        result = np.load(self.directory / f'{split}_indices.npy')
        if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer) or len(result) == 0:
            raise ValueError(f'Invalid or empty {split} indices')
        if result.min() < 0 or result.max() >= len(self.arr['counts']) or len(np.unique(result)) != len(result):
            raise ValueError(f'Out-of-range or duplicate {split} indices')
        return result

    def batch(self, ids):
        ids = np.asarray(ids)
        B = len(ids)
        gids_cpu = self.arr['graph_ids'][ids]
        N = int(self.arr['node_counts'][gids_cpu].max())
        idx = torch.tensor(ids, device=self.device)
        gids = self.gpu['graph_ids'][idx]
        n = self.gpu['node_counts'][gids]
        nm = torch.arange(N, device=self.device)[None, :] < n[:, None]
        X = F.one_hot(self.gpu['nodes'][gids, :N].long(), 12).float()[:, :, :11] * nm[:, :, None]
        E = F.one_hot(self.gpu['edges'][gids, :N, :N].long(), 5).float() * nm[:, :, None, None] * nm[:, None, :, None]
        y = torch.ones(B, 1, device=self.device)
        npeaks = self.gpu['peak_counts'][idx]
        valid = torch.arange(128, device=self.device)[None, :] < npeaks[:, None]
        form = self.lut[self.gpu['counts'][idx].long()].reshape(B, 128, 144) * valid[:, :, None]
        mask = torch.cat([torch.zeros(B, 1, device=self.device, dtype=torch.bool), ~valid], dim=1)
        sos = torch.zeros(B, 1, 13, device=self.device)
        sos[torch.arange(B, device=self.device), 0, self.gpu['adduct_ids'][idx]] = 1
        sos[torch.arange(B, device=self.device), 0, 2 + self.gpu['energy_bins'][idx].long()] = 1
        return sos, form, mask, X, E, y, nm


def train_plan(indices, steps, batch_size=64, seed=2026):
    if len(indices) == 0 or steps < 1:
        raise ValueError('Training needs nonempty indices and positive steps')
    generator = torch.Generator().manual_seed(seed)
    plan = np.full((steps, batch_size), -1, dtype=np.int64)
    step = 0
    while step < steps:
        order = indices[torch.randperm(len(indices), generator=generator).numpy()]
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            plan[step, :len(batch)] = batch
            step += 1
            if step == steps:
                break
    return plan


def monitor_indices(indices, ratio=0.1, seed=2026):
    count = int(ratio * len(indices))
    if count < 2:
        raise ValueError('Validation monitor must contain at least two spectra')
    generator = torch.Generator().manual_seed(seed)
    return indices[torch.randperm(len(indices), generator=generator).numpy()[:count]]


@torch.no_grad()
def evaluate(model, forward, data, ids, batch_size=64):
    rng, was_training = get_rng(), model.training
    model.eval()
    try:
        losses, ms, graph = [], [], []
        for offset in range(0, len(ids), batch_size):
            z = forward(*data.batch(ids[offset:offset + batch_size]))
            losses.append(finetune_loss(model, z).item())
            ms.append(F.normalize(z[0], dim=-1))
            graph.append(F.normalize(z[1], dim=-1))
        A, B = torch.cat(ms), torch.cat(graph)
        correct = 0
        for offset in range(0, len(ids), 256):
            predicted = A[offset:offset + 256].matmul(B.T).argmax(1)
            correct += (predicted == torch.arange(offset, min(offset + 256, len(ids)), device=A.device)).sum().item()
        return {'loss': float(np.mean(losses)), 'positive_cosine': (A * B).sum(-1).mean().item(),
                'row_index_top1': correct / len(ids), 'spectra': len(ids)}
    finally:
        model.train(was_training)
        set_rng(rng)
