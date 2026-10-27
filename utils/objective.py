"""EQ-COND training objective with the frozen native BFN algebra.

Combines formula conditioning, sender noise, time sampling, discrete Bayesian
updates, class reweighting and MS2Mol adapter/distillation terms. The full-batch
wrapper computes one global bond ratio and edge denominator, then multiplies
by ``world`` to compensate for DDP's gradient averaging.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch.nn import functional as F

from models.model import FormulaCondition


def _core(model):
    return model.module if hasattr(model, "module") else model


def counts(batches, device, *, distributed: bool = True) -> dict[str, float]:
    """Return exact logical-batch denominators over local micro-batches."""

    value = torch.tensor(
        [
            sum(len(batch.halfedge_type) for batch in batches),
            sum(int((batch.halfedge_type > 0).sum()) for batch in batches),
            sum(len(batch.node_type) for batch in batches),
            sum(len(batch.items) for batch in batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value)
    if value[0] <= 0 or value[2] <= 0 or value[3] <= 0:
        raise ValueError("Empty EQ-COND logical batch")
    return {
        "edges": float(value[0]),
        "bonds": float(value[1]),
        "nodes": float(value[2]),
        "graphs": float(value[3]),
    }


def _teacher_forward(teacher, *, cond_emb, formula_counts, kwargs):
    """Call a teacher with or without formula-count conditioning.

    An MS2Mol run may use an EQ-COND teacher, while the frozen
    Graph2Mol teacher has no ``formula_counts`` keyword.  The fallback is only
    for that explicit legacy signature; no alternate data path is accepted.
    """

    try:
        return teacher(cond_emb_cached=cond_emb, formula_counts=formula_counts, **kwargs)
    except TypeError as error:
        if "formula_counts" not in str(error):
            raise
        return teacher(cond_emb_cached=cond_emb, **kwargs)


def objective(
    model,
    batch,
    *,
    formula_counts: torch.Tensor,
    stage: str,
    step: int,
    seed: int = 2026,
    teacher=None,
    settings: Optional[Mapping[str, Any]] = None,
    t: Optional[torch.Tensor] = None,
    theta: Optional[torch.Tensor] = None,
):
    """Run one micro-batch forward and return streams for global reduction."""

    settings = dict(settings or {})
    if stage not in ("graph2mol", "ms2mol"):
        raise ValueError(f"Unsupported EQ-COND stage: {stage!r}")
    m = _core(model)
    dtype = m.node_embedder.weight.dtype
    device = m.node_embedder.weight.device
    batch.to(device, dtype)
    formula_counts = FormulaCondition.coerce_counts(
        formula_counts, device=device, dtype=dtype,
    )
    if formula_counts.shape[0] != len(batch.items):
        raise ValueError(
            "formula_counts must have one row per logical graph in the micro-batch"
        )

    labels = batch.halfedge_type.to(device)
    zmol = batch.zmol_target.to(device=device, dtype=dtype)

    # Share the stateless noise implementation with sampling.
    from utils.randomness import ManagedNoise

    noise = ManagedNoise(batch, seed)
    t = noise.time(step, device, dtype) if t is None else t.to(device=device, dtype=dtype)
    sender = noise.edge(
        step, (len(labels), m.num_edge_types), device, dtype, "train_sender"
    )
    theta = (
        m.discrete_bayesian_update(
            t, labels, batch.halfedge_type_batch, sender
        )
        if theta is None
        else theta.to(device=device, dtype=dtype)
    )
    node_noise = noise.node(
        step, (len(batch.node_type), m.node_dim), device, dtype, "train_node"
    )

    m.capture_layer_h_nodes = bool(stage == "ms2mol" and teacher is not None)
    m.use_zms_adapter = stage == "ms2mol"
    kwargs = dict(
        node_types=batch.node_type,
        edge_index=batch.halfedge_index,
        batch_node=batch.node_type_batch,
        batch_edge=batch.halfedge_type_batch,
        instrument_type_idx=batch.instrument_type_idx_batch,
        ionization_type_idx=batch.ionization_type_idx_batch,
        t=t,
        edge_theta=theta,
        edge_types_t=torch.zeros_like(labels),
        node_noise=node_noise,
    )
    if settings.get("student_condition_mode", "real") == "default":
        kwargs["instrument_type_idx"] = torch.full_like(
            kwargs["instrument_type_idx"], 2
        )
        kwargs["ionization_type_idx"] = torch.zeros_like(
            kwargs["ionization_type_idx"]
        )
    probs = model(
        cond_emb_cached=batch.cond_emb_cached,
        formula_counts=formula_counts,
        **kwargs,
    )
    student_logits = m._last_x0_logits
    student_layers = m._last_layer_h_nodes
    if stage == "ms2mol":
        if teacher is None:
            raise RuntimeError("MS2Mol objective requires an explicit teacher")
        teacher.eval()
        teacher.use_zms_adapter = False
        teacher.capture_layer_h_nodes = True
        teacher_kwargs = dict(kwargs)
        # The frozen Graph2Mol teacher is evaluated with its committed
        # instrument/adduct condition (2/0).  Student MS2Mol remains real
        # acquisition by default; callers must select any other mode explicitly.
        if settings.get("teacher_condition_mode", "default") == "default":
            teacher_kwargs["instrument_type_idx"] = torch.full_like(
                kwargs["instrument_type_idx"], 2
            )
            teacher_kwargs["ionization_type_idx"] = torch.zeros_like(
                kwargs["ionization_type_idx"]
            )
        with torch.no_grad():
            _teacher_forward(
                teacher,
                cond_emb=zmol,
                formula_counts=formula_counts,
                kwargs=teacher_kwargs,
            )

    if not torch.isfinite(probs).all():
        raise FloatingPointError("Non-finite EQ-COND edge probabilities")
    streams = {
        "t": t,
        "theta": theta,
        "sender_noise": sender,
        "node_noise": node_noise,
        "labels": labels,
        "student_logits": student_logits,
        "student_layers": student_layers,
    }
    return probs, streams


def fullbatch_objective(
    model,
    batch,
    *,
    formula_counts: torch.Tensor,
    stage: str,
    step: int,
    stats: Mapping[str, float],
    world: int = 1,
    seed: int = 2026,
    teacher=None,
    settings: Optional[Mapping[str, Any]] = None,
):
    """Return the exact logical-batch loss for one streamed micro-batch."""

    settings = dict(settings or {})
    m = _core(model)
    probs, streams = objective(
        model,
        batch,
        formula_counts=formula_counts,
        stage=stage,
        step=step,
        seed=seed,
        teacher=teacher,
        settings=settings,
    )
    labels = streams["labels"]
    one_hot = F.one_hot(labels, m.num_edge_types).to(probs.dtype)
    t_edge = streams["t"][batch.halfedge_type_batch]
    ratio = float(stats["bonds"]) / float(stats["edges"])
    weights = 1.0 + (1.0 - ratio) * (labels > 0).to(probs.dtype)
    bfn_per_edge = (
        m.num_edge_types
        * m.beta1
        * t_edge
        * ((one_hot - probs) ** 2).sum(dim=-1)
        * weights
    )
    loss = bfn_per_edge.sum() / float(stats["edges"])
    parts = {"bfn": loss}

    if stage == "ms2mol":
        if teacher is None:
            raise RuntimeError("MS2Mol objective requires an explicit teacher")
        adapter_mse = F.mse_loss(
            m._last_adapter_out,
            batch.zmol_target.to(device=probs.device, dtype=probs.dtype),
            reduction="sum",
        ) / (float(stats["graphs"]) * 512.0)
        teacher_logits = getattr(teacher, "_last_x0_logits", None)
        teacher_layers = getattr(teacher, "_last_layer_h_nodes", None)
        if teacher_logits is None or teacher_layers is None:
            raise RuntimeError("Teacher did not expose all six feature layers")
        temperature = float(settings.get("temperature", 2.0))
        kd = F.kl_div(
            F.log_softmax(m._last_x0_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="sum",
        ) * (temperature * temperature) / float(stats["edges"])
        student_layers = getattr(m, "_last_layer_h_nodes", None)
        if student_layers is None or len(student_layers) != 6 or len(teacher_layers) != 6:
            raise RuntimeError("EQ-COND featureKD requires all six node layers")
        feature_kd = sum(
            F.mse_loss(student_layers[i], teacher_layers[i], reduction="sum")
            / (float(stats["nodes"]) * float(m.hidden_dim))
            for i in range(6)
        ) / 6.0
        loss = (
            loss
            + float(settings.get("mse_weight", 0.5)) * adapter_mse
            + float(settings.get("weight", 1.0)) * kd
            + float(settings.get("feature_weight", 0.5)) * feature_kd
        )
        parts.update(adapter_mse=adapter_mse, kd=kd, feature_kd=feature_kd)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite EQ-COND logical full-batch loss")
    # DDP averages gradients across ranks.  Compensate exactly once per
    # micro-batch so the logical global objective is preserved.
    return loss * int(world), {key: float(value.detach()) for key, value in parts.items()}, streams


def train_step(
    model,
    batch,
    formula_counts,
    optimizer,
    *,
    stage: str,
    step: int,
    stats: Optional[Mapping[str, float]] = None,
    world: int = 1,
    seed: int = 2026,
    teacher=None,
    settings: Optional[Mapping[str, Any]] = None,
    grad_clip_norm: float = 50.0,
):
    """Single-batch engineering helper; full trainer streams micro-batches."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if stats is None:
        stats = counts([batch], batch.node_type.device, distributed=False)
    loss, details, streams = fullbatch_objective(
        model,
        batch,
        formula_counts=formula_counts,
        stage=stage,
        step=step,
        stats=stats,
        world=world,
        seed=seed,
        teacher=teacher,
        settings=settings,
    )
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if not torch.isfinite(grad):
        raise FloatingPointError("Non-finite EQ-COND gradient")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Non-finite EQ-COND parameter after optimizer")
    return loss, details, streams, grad
