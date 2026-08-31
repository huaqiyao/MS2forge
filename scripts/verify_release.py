#!/usr/bin/env python3
"""Verify that the release files, code architecture and checkpoints agree."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model import FLASH, GraphEncoder, MSEncoder  # noqa: E402


EXPECTED_FILES = {
    "checkpoints/align.pt": {
        "sha256": "a29c443dde401cbd97911fa7e56a835cc156af7768346febbace6317b8d940db",
        "size": 151822020,
    },
    "checkpoints/graph2mol_iter1460000.pt": {
        "sha256": "bceafa0bdc8cd28cfa58425da697cfff3f0d1a4df1b0c7d3e888d560e97a4c1f",
        "size": 455385676,
    },
    "checkpoints/ms2mol_iter80000.pt": {
        "sha256": "a77bd0fa4dcd41488a04dc1e4bd7bc780bc837aba6eaa0dbe46f2efb4b082778",
        "size": 464827486,
    },
    "data/cache/zms_v1.pt": {
        "sha256": "5b2e37cbda6e6ef8f70e66c2d6e5b5570e0449ad08d7f30e8f1af02a12fc247c",
        "size": 304977558,
        "large_cache": True,
    },
    "data/cache/zmol_v1.pt": {
        "sha256": "10aeb77702f9cfb1e7392d4ddeeb9c2fdbcf5664427be684824f3c22f07179c5",
        "size": 4463726086,
        "large_cache": True,
    },
    "data/cache/zmol_v1.pt.ready": {
        "sha256": "0e0137d662fbf47c1d38f7bb43de90cf589517aebb3b487517054df1a8f63457",
        "size": 18,
    },
}

LEGACY_EDGE_PRECISION_KEYS = {
    "edge_precision_head.0.weight",
    "edge_precision_head.0.bias",
    "edge_precision_head.2.weight",
    "edge_precision_head.2.bias",
}
ADAPTER_KEYS = {
    "zms_adapter.0.weight",
    "zms_adapter.0.bias",
    "zms_adapter.3.weight",
    "zms_adapter.3.bias",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_prefix(state, prefix: str):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def check_exact_keys(actual, expected, label: str):
    actual_set = set(actual)
    if actual_set != set(expected):
        raise AssertionError(
            f"{label}: expected {sorted(expected)}, got {sorted(actual_set)}"
        )


def verify_file_hashes(skip_large_cache_hash: bool):
    report = {}
    for relative, expected in EXPECTED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing release file: {relative}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise AssertionError(f"{relative}: size {size} != {expected['size']}")
        skipped = bool(skip_large_cache_hash and expected.get("large_cache"))
        digest = None if skipped else sha256(path)
        if digest is not None and digest != expected["sha256"]:
            raise AssertionError(f"{relative}: SHA256 mismatch")
        report[relative] = {
            "size": size,
            "sha256": expected["sha256"],
            "hash_checked": not skipped,
        }
        print(f"[OK] file {relative}" + (" (large hash skipped)" if skipped else ""))
    return report


def verify_align_checkpoint():
    path = ROOT / "checkpoints/align.pt"
    checkpoint = load_checkpoint(path)
    assert checkpoint.get("stage") == "align"
    assert checkpoint.get("iteration") == 19000
    state = checkpoint["model"]

    ms_state = strip_prefix(state, "ms_encoder.")
    graph_state = strip_prefix(state, "graph_encoder.")
    ms_encoder = MSEncoder(
        dim_sos=13,
        dim_formula=144,
        hidden_dim=512,
        num_transformer_layers=3,
        nhead=8,
        output_dim=512,
        dropout=0.1,
        input_dropout=0.1,
        max_len=129,
    )
    graph_encoder = GraphEncoder(n_layers=4)
    ms_encoder.load_state_dict(ms_state, strict=True)
    graph_encoder.load_state_dict(graph_state, strict=True)
    print("[OK] align.pt: both encoder substates strict-load")

    result = {
        "stage": "align",
        "iteration": 19000,
        "ms_encoder_tensors": len(ms_state),
        "graph_encoder_tensors": len(graph_state),
        "strict_load": True,
    }
    del checkpoint, state, ms_state, graph_state, ms_encoder, graph_encoder
    gc.collect()
    return result


def model_from_config(config_name: str):
    with (ROOT / "configs" / config_name).open("r", encoding="utf-8") as stream:
        config = EasyDict(yaml.safe_load(stream))
    atomic_numbers = list(config.chem.atomic_numbers)
    num_node_types = len(atomic_numbers) + 1
    num_edge_types = len(config.chem.mol_bond_types) + 1
    return FLASH(config.model, num_node_types, num_edge_types, atomic_numbers), config


def verify_graph_checkpoint():
    path = ROOT / "checkpoints/graph2mol_iter1460000.pt"
    checkpoint = load_checkpoint(path)
    assert checkpoint.get("stage") == "graph2mol"
    assert checkpoint.get("iteration") == 1460000
    model, config = model_from_config("train_graph2mol.yml")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    check_exact_keys(missing, set(), "graph checkpoint missing keys")
    check_exact_keys(unexpected, LEGACY_EDGE_PRECISION_KEYS, "graph checkpoint legacy keys")
    assert config.model.gat.hidden_dim == 512 and config.model.gat.num_layers == 6
    assert not any("lora" in key.lower() for key in checkpoint["model"])
    print("[OK] graph2mol checkpoint: current model matches; four disabled legacy-head keys whitelisted")
    result = {
        "stage": "graph2mol",
        "iteration": 1460000,
        "missing_keys": list(missing),
        "whitelisted_unexpected_keys": sorted(unexpected),
        "lora_keys": 0,
    }
    del checkpoint, model, config
    gc.collect()
    return result


def verify_ms_checkpoint():
    path = ROOT / "checkpoints/ms2mol_iter80000.pt"
    checkpoint = load_checkpoint(path)
    assert checkpoint.get("stage") == "ms2mol"
    assert checkpoint.get("iteration") == 80000
    model, config = model_from_config("train_ms2mol.yml")
    model.load_state_dict(checkpoint["model"], strict=True)
    actual_adapter = {key for key in checkpoint["model"] if key.startswith("zms_adapter.")}
    check_exact_keys(actual_adapter, ADAPTER_KEYS, "MS2Mol adapter keys")
    assert not any("lora" in key.lower() for key in checkpoint["model"])
    assert config.train.student_condition_mode == "real"
    assert config.train.teacher_condition_mode == "real"
    print("[OK] ms2mol checkpoint: strict-load; Adapter present; no LoRA")
    result = {
        "stage": "ms2mol",
        "iteration": 80000,
        "strict_load": True,
        "adapter_keys": sorted(actual_adapter),
        "lora_keys": 0,
        "student_condition_mode": "real",
        "teacher_condition_mode": "real",
    }
    del checkpoint, model, config
    gc.collect()
    return result


def verify_cross_stage_loading():
    graph_checkpoint = load_checkpoint(ROOT / "checkpoints/graph2mol_iter1460000.pt")
    model, _ = model_from_config("train_ms2mol.yml")
    missing, unexpected = model.load_state_dict(graph_checkpoint["model"], strict=False)
    check_exact_keys(missing, ADAPTER_KEYS, "graph -> MS2Mol missing keys")
    check_exact_keys(unexpected, LEGACY_EDGE_PRECISION_KEYS, "graph -> MS2Mol legacy keys")
    print("[OK] graph2mol -> MS2Mol initialization has the exact expected key transition")
    result = {"missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected)}
    del graph_checkpoint, model
    gc.collect()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-large-cache-hash",
        action="store_true",
        help="check cache presence/size but skip hashing zms_v1.pt and zmol_v1.pt",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "status": "running",
        "files": verify_file_hashes(args.skip_large_cache_hash),
        "checkpoints": {
            "align": verify_align_checkpoint(),
            "graph2mol": verify_graph_checkpoint(),
            "ms2mol": verify_ms_checkpoint(),
            "graph_to_ms2mol": verify_cross_stage_loading(),
        },
    }
    report["status"] = "passed"
    if args.json_output is not None:
        output = args.json_output if args.json_output.is_absolute() else ROOT / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] wrote {output}")
    print("\nRelease verification PASSED")


if __name__ == "__main__":
    main()
