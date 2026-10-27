"""Formula-only EQ-COND sampling with explicit node/edge noise mappings."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from models.model import build_eq_cond_model, FormulaCondition
from utils.checkpoint import read_weights
from utils.config import load_yaml
from utils.eval_utils import build_candidate_cache_record
from utils.graph import StrictFormulaDataset, FailedInput, collate_generation
from utils.packed import ZmsMmapCache
from utils.randomness import ManagedNoise
from utils.artifacts import resolve, encoder_identity, require_encoder
from utils.compat import normalize_model_config
from utils.permutation import relabel


RECONSTRUCTION_PROTOCOL = "formula_slot_reconstruction_v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def completed(path):
    """Discard only an interrupted final record, never an interior error."""
    found = set()
    if not path.exists():
        return found
    with path.open("r+b") as f:
        size = path.stat().st_size
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                f.truncate(offset)
                break
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                if f.tell() != size:
                    raise ValueError("Corrupt interior record in candidate output")
                f.truncate(offset)
                break
            sid = record["spec_id"]
            if sid in found:
                raise ValueError(f"Duplicate completed input: {sid}")
            found.add(sid)
    return found


def reconstruct_candidates(item, predictions, probabilities, atomic_numbers):
    """Restore formula-slot indexing before RDKit's order-sensitive sanitization.

    The node/edge maps come from input construction, never from a reference
    structure. Bond labels and probability rows follow the same inverse map.
    Full sanitization and connectedness checks remain in the candidate builder.
    """
    mapping = relabel(item.node_to_base)
    if not torch.equal(item.halfedge_index.cpu(), mapping.edges):
        raise ValueError('Generation edges do not match the complete unordered graph')
    if not torch.equal(item.edge_to_base.cpu(), mapping.new_to_old_edge):
        raise ValueError('Generation node/edge permutation mappings disagree')
    if len(predictions) != len(probabilities):
        raise ValueError('Prediction and probability trajectory counts disagree')
    nodes = item.node_type.cpu().index_select(0, mapping.old_to_new_node)
    edge_order = mapping.old_to_new_edge
    labels, probs = [], []
    for prediction, probability in zip(predictions, probabilities):
        if prediction.shape != (len(edge_order),) or probability.ndim != 2 or probability.shape[0] != len(edge_order):
            raise ValueError('Prediction/probability rows do not match candidate pairs')
        labels.append(prediction.cpu().index_select(0, edge_order))
        probs.append(probability.cpu().index_select(0, edge_order))
    record = build_candidate_cache_record(
        labels, None, node_types=nodes, halfedge_index=mapping.edges,
        atomic_numbers=atomic_numbers, pred_probs=probs)
    record.pop('true')
    record['reconstruction_protocol'] = RECONSTRUCTION_PROTOCOL
    return record


def generate(model, items, cfg, device):
    sequences = [[] for _ in items]
    probabilities = [[] for _ in items]
    formula = torch.stack([FormulaCondition.from_formula(x.formula.text) for x in items])
    for draw in range(cfg["draws"]):
        batch = collate_generation([replace(x, draw_id=draw) for x in items]).to(device)
        labels, probs = model.sample_bfn(
            node_types=batch.node_type, edge_index=batch.halfedge_index,
            batch_node=batch.node_type_batch, batch_edge=batch.halfedge_type_batch,
            instrument_type_idx=batch.instrument_type_idx_batch,
            ionization_type_idx=batch.ionization_type_idx_batch,
            cond_emb_cached=batch.cond_emb_cached, formula_counts=formula,
            n_timesteps=cfg["steps"], noise_provider=ManagedNoise(batch, cfg["seed"]),
            return_probabilities=True, disable_tqdm=True)
        for i in range(len(items)):
            a, b = batch.edge_offsets[i:i + 2]
            sequences[i].append(labels[a:b].cpu())
            probabilities[i].append(probs[a:b].float().cpu())
    for item, preds, probs in zip(items, sequences, probabilities):
        record = reconstruct_candidates(item, preds, probs, model.atomic_numbers)
        record.update(spec_id=item.sample_id, formula=item.formula.text,
                      architecture_id=model.architecture_id)
        yield record


def main(dataset=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["msg", "nplib1"], help="Override dataset in the YAML")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = load_yaml(f, dataset=args.dataset or dataset)
    if dataset and cfg["dataset"] != dataset:
        p.error("Dataset entry point and config disagree")
    rank, world, local_rank = (int(os.environ.get(k, default))
                               for k, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if args.device not in ("auto", "cpu") and world == 1:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if min(cfg["draws"], cfg["steps"], cfg["batch_size"]) < 1:
        p.error("draws, steps and batch_size must be positive")
    weight = resolve(args.checkpoint or cfg["checkpoint"], 'ms2mol')
    payload, state = read_weights(weight)
    model = build_eq_cond_model(cfg, stage="ms2mol", device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    meta = json.loads((Path(cfg["zms_root"]) / "CACHE_MANIFEST.json").read_text())
    expected = encoder_identity(cfg)
    if expected and meta.get("encoder_sha256") != expected:
        raise ValueError("Zms cache came from a different encoder")
    if not meta.get('encoder_sha256'):
        raise ValueError('Zms cache is missing encoder identity')
    require_encoder(payload, meta['encoder_sha256'], 'MS2Mol sampler')
    inputs = StrictFormulaDataset.from_file(cfg["inputs"], ZmsMmapCache(cfg["zms_root"]),
                                           adapter=cfg["dataset"], order_seed=None)
    ids = [str(row["sample_id"]) for row in inputs.rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate generation sample IDs")
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    target, manifest = out / f"candidates_rank{rank}.jsonl", out / f"run_rank{rank}.json"
    identity = {"config": cfg, "checkpoint_sha256": sha256(weight), "rank": rank, "world": world,
                "reconstruction_protocol": RECONSTRUCTION_PROTOCOL,
                "inputs_sha256": sha256(cfg["inputs"]), "encoder_sha256": meta['encoder_sha256'],
                "zms_manifest_sha256": sha256(Path(cfg['zms_root']) / 'CACHE_MANIFEST.json')}
    if manifest.exists():
        saved_identity = json.loads(manifest.read_text())
        if "model" in saved_identity.get("config", {}):
            saved_identity["config"]["model"] = normalize_model_config(saved_identity["config"]["model"])
        if not args.resume or saved_identity != identity:
            raise ValueError("Existing output requires --resume with matching inputs/config/weights/world")
    elif target.exists():
        raise ValueError("Candidate file has no matching run identity")
    else:
        manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    done = completed(target) if args.resume else set()
    pending, edges = [], 0
    with target.open("a", encoding="utf-8") as f:
        def emit(record):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        def flush():
            for record in generate(model, pending, cfg, device):
                emit(record)

        for index in range(rank, len(inputs), world):
            if ids[index] in done:
                continue
            item = inputs[index]
            if isinstance(item, FailedInput):
                emit({"type": "input_failure", "spec_id": item.sample_id, "reason": item.reason})
                continue
            n_edges = item.halfedge_index.shape[1]
            if pending and (len(pending) >= cfg["batch_size"]
                            or edges + n_edges > cfg.get("max_halfedges", 12000)):
                flush()
                pending, edges = [], 0
            pending.append(item)
            edges += n_edges
        if pending:
            flush()


if __name__ == "__main__":
    main()
