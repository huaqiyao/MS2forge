#!/usr/bin/env python3
"""Run a checkpoint-free MS2Forge forward-pass smoke test.

The demo loads the released sampling architecture, builds a tiny synthetic
molecular graph, and checks that FLASH returns finite bond probabilities.
It intentionally needs neither the dataset nor a model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model import FLASH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-free forward smoke test for MS2Forge."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sample.yml",
        help="Model config to instantiate (default: configs/sample.yml).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: CUDA when available).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def complete_directed_edges(num_nodes: int, device: torch.device) -> torch.Tensor:
    pairs = [(source, target) for source in range(num_nodes) for target in range(num_nodes)
             if source != target]
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    with args.config.open("r", encoding="utf-8") as stream:
        config = EasyDict(yaml.safe_load(stream))

    atomic_numbers = list(config.chem.atomic_numbers)
    num_node_types = len(atomic_numbers) + 1
    num_edge_types = len(config.chem.mol_bond_types) + 1
    model = FLASH(
        config.model,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        atomic_numbers=atomic_numbers,
    ).to(device)
    model.eval()

    # One four-atom graph with all directed, non-self edges.
    num_nodes = 4
    edge_index = complete_directed_edges(num_nodes, device)
    num_edges = edge_index.shape[1]
    node_types = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    batch_node = torch.zeros(num_nodes, dtype=torch.long, device=device)
    batch_edge = torch.zeros(num_edges, dtype=torch.long, device=device)
    instrument_type_idx = torch.tensor([0], dtype=torch.long, device=device)
    ionization_type_idx = torch.tensor([0], dtype=torch.long, device=device)
    time = torch.tensor([0.5], dtype=torch.float32, device=device)
    edge_types_t = torch.zeros(num_edges, dtype=torch.long, device=device)
    cond_emb = torch.zeros(
        (1, int(config.model.contrastive_dim)), dtype=torch.float32, device=device
    )

    with torch.inference_mode():
        probabilities = model(
            node_types=node_types,
            edge_index=edge_index,
            batch_node=batch_node,
            batch_edge=batch_edge,
            instrument_type_idx=instrument_type_idx,
            ionization_type_idx=ionization_type_idx,
            t=time,
            edge_types_t=edge_types_t,
            cond_emb_cached=cond_emb,
        )

    expected_shape = (num_edges, num_edge_types)
    if tuple(probabilities.shape) != expected_shape:
        raise AssertionError(
            f"unexpected output shape: {tuple(probabilities.shape)} != {expected_shape}"
        )
    if not torch.isfinite(probabilities).all():
        raise AssertionError("model output contains NaN or Inf")
    row_sums = probabilities.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise AssertionError("bond probabilities do not sum to one")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "passed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config": str(args.config.resolve()),
        "stage": str(config.model.stage),
        "parameter_count": parameter_count,
        "nodes": num_nodes,
        "directed_edges": num_edges,
        "output_shape": list(probabilities.shape),
        "probability_row_sum_min": float(row_sums.min().item()),
        "probability_row_sum_max": float(row_sums.max().item()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
