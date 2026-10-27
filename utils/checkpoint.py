"""Dynamic EQ-COND state_dict exports and optional trainer checkpoints."""
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from utils.artifacts import resolve, require_encoder, publish
from utils.compat import normalize_model_config

ARCHITECTURE = "eq_cond_unordered_bfn_v1"


def read_weights(path):
    source = Path(path)
    path = resolve(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a checkpoint or tensor state_dict")
    if "architecture_id" in payload and payload["architecture_id"] != ARCHITECTURE:
        raise ValueError("Checkpoint architecture differs from EQ-COND")
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint contains no tensor state_dict")
    if not any(k.startswith("formula_condition.") for k in state):
        raise ValueError("Legacy checkpoint: missing EQ-COND formula/FiLM parameters")
    if any("bond_ffn_left" in k or "bond_ffn_right" in k for k in state):
        raise ValueError("Directional edge parameters are not compatible with EQ-COND")
    # Keep lineage outside the tensor-only .pt file. Direct paths and published
    # pointers both retain encoder verification, including the physical hash.
    sidecar = path.with_name(path.name + ".json")
    metadata_path = source if source.suffix == ".json" else sidecar
    if metadata_path.is_file():
        if resolve(metadata_path) != path:
            raise ValueError("Checkpoint metadata points to a different weight file")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded = metadata.get("encoder_sha256")
        embedded = payload.get("encoder_sha256") or payload.get("config", {}).get("encoder_sha256")
        if recorded and embedded and recorded != embedded:
            raise ValueError("Checkpoint metadata encoder identity mismatch")
        if recorded:
            payload = {"model": state} if state is payload else dict(payload)
            payload["encoder_sha256"] = recorded
    return payload, state


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save(path, model, optimizer, scheduler, sampler, config, update, rank, world):
    path = Path(path)
    sidecar = path.with_name(path.name + ".json")
    if path.exists() or path.with_suffix(".partial").exists() or sidecar.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    fields = config.get("train", {}).get("checkpoint_fields", ["model"])
    # Optional fields are enabled by uncommenting checkpoint_fields in the YAML.
    # Lazy getters ensure disabled optimizer/RNG/training states are not collected.
    getters = {
        "model": model.state_dict,
        "optimizer": lambda: optimizer.state_dict(),
        "scheduler": lambda: scheduler.state_dict(),
        "sampler": lambda: sampler.state_dict(),
        "rng": rng_state,
        "config": lambda: config,
        "update": lambda: update,
        "schema": lambda: "ms2forge.eq_portable.v1",
        "architecture_id": lambda: ARCHITECTURE,
        "stage": lambda: config["stage"],
        "rank": lambda: rank,
        "world": lambda: world,
        "encoder_sha256": lambda: config.get("encoder_sha256"),
    }
    if (not isinstance(fields, list) or not all(isinstance(k, str) for k in fields)
            or "model" not in fields or len(set(fields)) != len(fields)
            or set(fields) - getters.keys()):
        raise ValueError("train.checkpoint_fields must list unique supported fields including model")
    # Match the released Graph2Mol file exactly: parameter names at the top level.
    # state_dict() automatically includes buffers and MS2Mol's zms_adapter layers.
    payload = model.state_dict() if fields == ["model"] else {k: getters[k]() for k in fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if config.get("encoder_sha256"):
        publish(sidecar, path, config["stage"], encoder_sha256=config["encoder_sha256"])


def restore(path, model, optimizer, scheduler, sampler, config, rank, world):
    payload, state = read_weights(path)
    required = {"optimizer", "scheduler", "sampler", "rng", "config", "update", "rank", "world", "stage"}
    if payload.get("schema") != "ms2forge.eq_portable.v1" or not required <= payload.keys():
        raise ValueError("Exact resume requires all optional checkpoint_fields; use --init for weights only")
    require_encoder(payload, config.get('encoder_sha256'), 'Resume')
    if (payload["rank"], payload["world"], payload["stage"]) != (rank, world, config["stage"]):
        raise ValueError("Resume rank/world/stage mismatch")
    for key in ("model", "data", "kd", "optimizer", "scheduler"):
        saved_value, current_value = payload["config"].get(key), config.get(key)
        if key == "model":
            saved_value = normalize_model_config(saved_value)
            current_value = normalize_model_config(current_value)
        if saved_value != current_value:
            raise ValueError(f"Resume changes {key}")
    for key in ("seed", "global_batch", "max_micro_graphs", "max_micro_halfedges", "warmup_updates"):
        if payload["config"]["train"][key] != config["train"][key]:
            raise ValueError(f"Resume changes train.{key}")
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(payload["update"])
