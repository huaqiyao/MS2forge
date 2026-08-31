#!/usr/bin/env python3
"""Run NEO CASMI sampling with formula-only inputs and sample-stable RNG.

The upstream sampler seeds by distributed rank.  For a paper benchmark that
means changing the number of GPUs can change a sample's candidates.  This
wrapper requires batch_size=1 and reseeds each stochastic sampling chunk from
SHA256(global_seed, spectrum_id, chunk_index), making predictions independent
of DDP rank and shard composition.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sample as base_sample

from casmi_dataset import CASMIDiffMSDataset


_STATE = {"global_seed": None, "spec_id": None, "chunk_index": 0}
_CONTROL = os.environ.get("CASMI_CONDITION_CONTROL", "real").strip().lower()
if _CONTROL not in {"real", "zero", "shuffle"}:
    raise ValueError(f"Unsupported CASMI_CONDITION_CONTROL={_CONTROL!r}")
_SHUFFLE_MAP = None
if _CONTROL == "shuffle":
    mapping_path = os.environ.get("CASMI_SHUFFLE_MAP_JSON")
    if not mapping_path:
        raise ValueError("CASMI_SHUFFLE_MAP_JSON is required for shuffle control")
    with open(mapping_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    _SHUFFLE_MAP = payload.get("mapping", payload)


def argument_value(flag: str) -> Optional[str]:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def resolve_global_seed() -> int:
    explicit = argument_value("--seed")
    if explicit is not None:
        return int(explicit)
    config_path = argument_value("--config") or "./configs/sample.yml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return int(config["evaluate"]["seed"])


def stable_seed(global_seed: int, spec_id: str, chunk_index: int) -> int:
    payload = f"{global_seed}\0{spec_id}\0{chunk_index}".encode("utf-8")
    # Stay within the cross-library signed 31-bit seed range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


_ORIGINAL_MAKE_COLLATE = base_sample.make_msg_diffms_collate_with_cache


def deterministic_make_collate(*args, **kwargs):
    zms_cache = args[0] if args else kwargs.get("zms_cache")
    original_collate = _ORIGINAL_MAKE_COLLATE(*args, **kwargs)

    def collate(items):
        if len(items) != 1:
            raise RuntimeError(
                "Sample-stable RNG requires evaluate.batch_size=1; "
                f"received a batch of {len(items)}"
            )
        spec_id = getattr(items[0], "mol_id", None)
        if spec_id is None:
            raise RuntimeError("Dataset item lacks mol_id required for sample-stable RNG")
        _STATE["spec_id"] = str(spec_id)
        _STATE["chunk_index"] = 0
        batch = original_collate(items)
        if batch is None:
            return None
        if _CONTROL == "zero":
            batch.cond_emb_cached = torch.zeros_like(batch.cond_emb_cached)
        elif _CONTROL == "shuffle":
            donor_id = str(_SHUFFLE_MAP[str(spec_id)])
            if donor_id == str(spec_id) or donor_id not in zms_cache:
                raise RuntimeError(f"Invalid shuffled-spectrum donor {donor_id!r} for {spec_id!r}")
            batch.cond_emb_cached = zms_cache[donor_id].float().reshape(1, -1)
        batch.condition_control = _CONTROL
        return batch

    return collate


_ORIGINAL_SAMPLE_BFN = base_sample.FLASH.sample_bfn


def deterministic_sample_bfn(self, *args, **kwargs):
    if _STATE["global_seed"] is None or _STATE["spec_id"] is None:
        raise RuntimeError("Sample-stable RNG state was not initialized")
    chunk_index = int(_STATE["chunk_index"])
    seed = stable_seed(int(_STATE["global_seed"]), str(_STATE["spec_id"]), chunk_index)
    _STATE["chunk_index"] = chunk_index + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return _ORIGINAL_SAMPLE_BFN(self, *args, **kwargs)


if __name__ == "__main__":
    _STATE["global_seed"] = resolve_global_seed()
    base_sample.DiffMSMSGDataset = CASMIDiffMSDataset
    base_sample.make_msg_diffms_collate_with_cache = deterministic_make_collate
    base_sample.FLASH.sample_bfn = deterministic_sample_bfn
    print(
        "[sample-stable RNG] enabled: "
        "seed=SHA256(global_seed, spectrum_id, stochastic_chunk), batch_size must be 1, "
        f"condition_control={_CONTROL}"
    )
    base_sample.main()
    if int(os.environ.get("RANK", "0")) == 0:
        output_dir = argument_value("--output_dir")
        if output_dir:
            for name in os.listdir(output_dir):
                if not name.startswith("summary_") or not name.endswith(".json"):
                    continue
                path = os.path.join(output_dir, name)
                with open(path, encoding="utf-8") as handle:
                    summary = json.load(handle)
                summary["condition_control"] = _CONTROL
                if _CONTROL == "shuffle":
                    mapping_path = os.environ["CASMI_SHUFFLE_MAP_JSON"]
                    summary["shuffle_map"] = mapping_path
                    summary["shuffle_map_sha256"] = hashlib.sha256(
                        open(mapping_path, "rb").read()
                    ).hexdigest()
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
