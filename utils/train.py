"""EQ-COND training with global loss normalization and distributed data parallelism."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.model import build_eq_cond_model, FormulaCondition
from utils import checkpoint, objective
from utils.config import load_yaml
from utils.graph import collate_supervised
from utils.packed import PackedGraphDataset, PackedMSDataset, GlobalVisitSchedule
from utils.transfer import load_graph_parent
from utils.artifacts import encoder_identity, require_encoder, publish, sha256


class ObjectiveForward(nn.Module):
    """DDP must see the full BFN + Adapter + six-layer KD loss graph."""
    def __init__(self, student, teacher, settings):
        super().__init__()
        self.student = student
        # The frozen teacher is not part of DDP's parameter tree.
        object.__setattr__(self, "_teacher", teacher)
        self.settings = settings
        self.last_details = {}

    def forward(self, batch, formula_counts, *, stage, step, stats, world, seed):
        loss, self.last_details, _ = objective.fullbatch_objective(
            self.student, batch, formula_counts=formula_counts,
            stage=stage, step=step, stats=stats, world=world, seed=seed,
            teacher=self._teacher, settings=self.settings)
        return loss


def microbatches(dataset, tokens, stage, settings):
    current, formula, edges = [], [], 0
    for token in tokens:
        graph = dataset[token]
        count = len(graph.halfedge_type)
        if count > settings["max_micro_halfedges"]:
            raise ValueError(f"Graph exceeds half-edge budget: {graph.sample_id}")
        row = dataset.connection().execute(
            "SELECT formula FROM records WHERE idx=?", (int(token[0]),)).fetchone()
        if not row or not row[0]:
            raise ValueError("Supervised row is missing its neutral formula")
        if current and (len(current) >= settings["max_micro_graphs"]
                        or edges + count > settings["max_micro_halfedges"]):
            yield collate_supervised(current, stage), torch.stack(formula)
            current, formula, edges = [], [], 0
        current.append(graph)
        formula.append(FormulaCondition.from_formula(row[0], adapter="graph2mol"))
        edges += count
    if current:
        yield collate_supervised(current, stage), torch.stack(formula)


def validate(model, teacher, dataset, cfg, device, rank, world):
    model.eval()
    if teacher is not None:
        teacher.eval()
    # Compute full validation denominators without retaining every graph in memory.
    tokens = [(i, 0, i) for i in range(rank, len(dataset), world)]
    local = torch.zeros(4, dtype=torch.float64, device=device)
    for batch, _ in microbatches(dataset, tokens, cfg["stage"], cfg["train"]):
        local += torch.tensor([len(batch.halfedge_type), (batch.halfedge_type > 0).sum().item(),
                               len(batch.node_type), len(batch.items)], device=device)
    if world > 1:
        dist.all_reduce(local)
    stats = dict(zip(("edges", "bonds", "nodes", "graphs"), local.tolist()))
    if min(stats["edges"], stats["graphs"]) <= 0:
        raise ValueError("Empty validation set")
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch, formula in microbatches(dataset, tokens, cfg["stage"], cfg["train"]):
            value, _, _ = objective.fullbatch_objective(
                model, batch, formula_counts=formula, stage=cfg["stage"], step=0,
                stats=stats, world=1, seed=cfg["train"]["seed"], teacher=teacher, settings=cfg["kd"])
            total += value.detach().double()
    if world > 1:
        dist.all_reduce(total)
    return float(total)


def main(dataset=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["msg", "nplib1"], help="Override dataset in the YAML")
    p.add_argument("--parent", help="EQ Graph2Mol parent for the MS student and frozen teacher")
    p.add_argument("--resume", help="Full optional trainer checkpoint (not default weights-only); {rank} expands per process")
    p.add_argument("--init", help="Load EQ weights only, without restoring optimizer or update number")
    p.add_argument("--to", type=int, help="Stop at this absolute update")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.resume and args.init:
        p.error("--resume and --init are mutually exclusive")
    with open(args.config, encoding="utf-8") as f:
        cfg = load_yaml(f, dataset=args.dataset or dataset)
    cfg['encoder_sha256'] = encoder_identity(cfg)
    if dataset and cfg["dataset"] != dataset:
        p.error("Dataset entry point and config disagree")
    if cfg.get("architecture_id") != checkpoint.ARCHITECTURE:
        p.error("Expected an EQ-COND config")
    stage = cfg["stage"]
    if stage not in ("graph2mol", "ms2mol"):
        p.error("Expected graph2mol or ms2mol")
    rank, world, local_rank = (int(os.environ.get(k, default))
                               for k, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if args.device not in ("auto", "cpu"):
        device = torch.device(args.device) if world == 1 else device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = cfg["train"]["seed"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed + rank)

    def get_data(split):
        if stage == "graph2mol":
            return PackedGraphDataset(cfg["data"][split], seed)
        return PackedMSDataset(cfg["data"][split], seed,
                               zms_root=cfg["data"].get("zms_root"), zmol_root=cfg["data"].get("zmol_root"))

    train, val = get_data("train"), get_data("val")
    for split, item in (("train", train), ("val", val)):
        if item.manifest.get("role") != split:
            raise ValueError(f"Packed {split} cache has the wrong split role")
        expected = cfg.get("encoder_sha256")
        if expected and item.manifest.get("encoder_sha256") != expected:
            raise ValueError(f"Packed {split} cache came from a different encoder")
    if train.manifest.get("encoder_sha256") != val.manifest.get("encoder_sha256"):
        raise ValueError("Train and validation encoder identities differ")
    cfg['encoder_sha256'] = train.manifest.get('encoder_sha256')
    if not cfg['encoder_sha256']:
        raise ValueError('Training cache is missing encoder identity')
    if stage == "ms2mol":
        for dataset in (train, val):
            for tower in ("zms", "zmol"):
                root = cfg["data"].get(tower + "_root") or dataset.manifest[tower + "_root"]
                manifest = Path(root) / "CACHE_MANIFEST.json"
                meta = json.loads(manifest.read_text())
                if (meta.get("encoder_sha256") != train.manifest.get("encoder_sha256")
                        or meta.get("tower") != tower or meta.get("normalization") != "NONE"):
                    raise ValueError(f"{tower} cache identity or normalization mismatch")
                recorded = dataset.manifest.get(tower + '_manifest_sha256')
                if recorded and sha256(manifest) != recorded:
                    raise ValueError(f'{tower} cache changed since supervision was packed; rebuild packed inputs')
    batch_size = cfg["train"]["global_batch"]
    if len(train) < batch_size:
        raise ValueError("Training set must contain at least one unique global batch")
    student = build_eq_cond_model(cfg, stage=stage, device=device)
    teacher = None
    if stage == "ms2mol":
        parent_path = args.parent or cfg["parent"]
        payload, state = checkpoint.read_weights(parent_path)
        # Shared Graph2Mol transfer may cross encoder spaces. Keep the student's
        # cache identity; parent integrity and tensor compatibility are checked
        # by read_weights, strict teacher loading and load_graph_parent.
        parent_encoder = payload.get('encoder_sha256') or payload.get('config', {}).get('encoder_sha256')
        print(json.dumps(dict(event='graph2mol_transfer', parent=str(parent_path),
                              parent_encoder_sha256=parent_encoder,
                              encoder_sha256=cfg['encoder_sha256'])), flush=True)
        teacher = build_eq_cond_model(cfg, stage="graph2mol", device=device)
        teacher.load_state_dict(state, strict=True)
        teacher.eval().requires_grad_(False)
        load_graph_parent(student, state)  # all 371 parent tensors, including all 54 FiLM tensors
    if args.init:
        payload, state = checkpoint.read_weights(args.init)
        require_encoder(payload, cfg['encoder_sha256'], 'Initialization')
        student.load_state_dict(state, strict=True)
    sampler = GlobalVisitSchedule(len(train), batch_size, seed, rank, world)
    optimizer = torch.optim.AdamW(student.parameters(), **cfg["optimizer"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg["scheduler"])
    start = 0
    if args.resume:
        start = checkpoint.restore(args.resume.format(rank=rank), student, optimizer,
                                   scheduler, sampler, cfg, rank, world)
    target = args.to if args.to is not None else cfg["train"]["updates"]
    if target <= start:
        raise ValueError("Target update must be greater than the restored update")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    log = output / f"train_rank{rank}.jsonl"
    if log.exists() and not args.resume:
        raise FileExistsError("Existing training log: choose a new output or explicitly resume")
    wrapper = ObjectiveForward(student, teacher, cfg["kd"])
    model = DDP(wrapper, device_ids=[device.index] if use_cuda else None,
                find_unused_parameters=True, broadcast_buffers=False) if world > 1 else wrapper
    for update in range(start + 1, target + 1):
        tick = time.monotonic()
        model.train()
        if teacher is not None:
            teacher.eval()
        groups = list(microbatches(train, sampler.indices(), stage, cfg["train"]))
        stats = objective.counts([b for b, _ in groups], device, distributed=world > 1)
        if int(stats["graphs"]) != batch_size:
            raise ValueError("Logical batch graph count differs from global_batch")
        if update <= cfg["train"]["warmup_updates"]:
            for group in optimizer.param_groups:
                group["lr"] = cfg["optimizer"]["lr"] * update / cfg["train"]["warmup_updates"]
        optimizer.zero_grad(set_to_none=True)
        parts = dict.fromkeys(("bfn", "adapter_mse", "kd", "feature_kd"), 0.0)
        for i, (batch, formula) in enumerate(groups):
            context = model.no_sync() if world > 1 and i < len(groups) - 1 else contextlib.nullcontext()
            with context:
                loss = model(batch, formula, stage=stage, step=update, stats=stats, world=world, seed=seed)
                loss.backward()
            for k, value in wrapper.last_details.items():
                parts[k] += value
        grad = nn.utils.clip_grad_norm_(student.parameters(), cfg["train"]["grad_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient; update not committed")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in student.parameters()):
            raise FloatingPointError("Nonfinite parameter after update")
        sampler.commit()
        values = torch.tensor(list(parts.values()), device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(values)
        record = dict(update=update, **dict(zip(parts, values.tolist())),
                      grad_norm=float(grad), seconds=time.monotonic() - tick,
                      global_batch=batch_size, lr=optimizer.param_groups[0]["lr"])
        if update % cfg["train"]["validation_every"] == 0:
            record["val_loss"] = validate(student, teacher, val, cfg, device, rank, world)
            scheduler.step(record["val_loss"])
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if rank == 0 and (update == start + 1 or update % 100 == 0):
            print(json.dumps(record), flush=True)
        if update == target or update % cfg["train"]["checkpoint_every"] == 0:
            checkpoint.save(output / f"eq_{stage}_u{update}_rank{rank}.pt",
                            student, optimizer, scheduler, sampler, cfg, update, rank, world)
            if world > 1:
                dist.barrier()
            if rank == 0:
                selected = output / f"eq_{stage}_u{update}_rank0.pt"
                publish(output / 'latest.json', selected, stage,
                        encoder_sha256=cfg['encoder_sha256'], update=update)
                if cfg.get('publish_checkpoint'):
                    publish(cfg['publish_checkpoint'], selected, stage,
                            encoder_sha256=cfg['encoder_sha256'], update=update)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
