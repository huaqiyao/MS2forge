#!/usr/bin/env python3
"""Shared feature preparation utilities for MS2Forge reranking."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from rerank import training_eval as feval


DEFAULT_EXCLUDE = feval.ID_COLUMNS | {
    "candidate_is_true",
    "candidate_is_true_strict",
    "candidate_is_true_isomorphic",
    "candidate_is_true_isomorphic_nostereo",
    "true_rank",
    "true_rank_strict",
    "true_rank_isomorphic",
    "true_rank_isomorphic_nostereo",
    "is_true",
    "true_smiles",
    "true_canonical_smiles",
    "candidate_canonical_smiles",
    "candidate_canonical_smiles_nostereo",
    "candidate_smiles",
    "smiles",
    "inchi",
    "inchikey",
    "included_reason",
    "mass_frag_ionization",
}

BASE_FEATURES = [
    "candidate_rank_freq",
    "candidate_freq",
    "candidate_first_seen",
    "candidate_freq_fraction",
    "n_samples",
    "n_unique",
    # Final-MS2Forge rich outputs expose one aggregate per sampled edge set.
    # Keep the legacy bond-only names below for backwards-compatible runs,
    # but prefer these populated final-model statistics when available.
    "mean_top1_prob_mean",
    "mean_top1_prob_max",
    "mean_top1_prob_min",
    "mean_margin_mean",
    "mean_margin_max",
    "mean_margin_min",
    "mean_entropy_mean",
    "mean_entropy_max",
    "mean_entropy_min",
    "sum_selected_log_prob_mean",
    "sum_selected_log_prob_max",
    "sum_selected_log_prob_min",
    "mean_selected_log_prob_mean",
    "mean_selected_log_prob_max",
    "mean_selected_log_prob_min",
    "n_edges_mean",
    "n_edges_max",
    "n_edges_min",
    "mean_margin_bond_mean",
    "mean_entropy_bond_mean",
    "mean_logp_bond_mean",
    "pred_bond_count_mean",
    "aromatic_count_mean",
    "num_atoms",
    "num_bonds",
    "ring_count",
    "aromatic_bond_count",
    "rdkit_valid",
    "log_freq",
    "inv_freq_rank",
    "freq_rank",
    "has_bfn_stats_num",
]

Z_FEATURES = [
    "z_available",
    "zms_zmol_cos",
    "z_zms_zmol_cos",
    "score_freq",
    "score_v0",
    "score_freq_plus_0p1z",
    "score_v0_plus_0p1z",
    "score_freq_plus_0p3z",
    "score_v0_plus_0p3z",
    "score_freq_plus_1p0z",
    "score_v0_plus_1p0z",
    "score_freq_plus_3p0z",
    "score_v0_plus_3p0z",
]

FRAG_FEATURES = [
    "mass_frag_parent_mz",
    "mass_frag_parent_exact_mass",
    "mass_frag_num_neutral_fragments",
    "mass_frag_num_direct_mz",
    "mass_frag_num_neutral_loss_mz",
    "mass_frag_num_observed_peaks",
    "mass_frag_num_matched_peaks",
    "mass_frag_direct_hits",
    "mass_frag_neutral_loss_hits",
    "mass_frag_explained_intensity",
    "mass_frag_top_peak_coverage",
    "mass_frag_match_fraction",
    "score_fragment_proxy",
    "score_mass_fragment_proxy",
    "mass_frag_available",
]

FRAG_V2_FEATURES = [
    "mass_frag_v2_available",
    "mass_frag_v2_top_intensity_explained",
    "mass_frag_v2_parent_ppm_error",
    "mass_frag_v2_parent_mass_score",
    "mass_frag_v2_common_loss_hits",
    "mass_frag_v2_common_loss_weighted_hits",
    "mass_frag_v2_candidate_common_loss_support",
    "mass_frag_v2_candidate_common_loss_observed",
    "mass_frag_v2_fragment_density",
    "score_fragment_proxy_v2",
    "score_mass_fragment_proxy_v2",
]

FEATURE_PRESETS = {
    "base": BASE_FEATURES,
    "base_z": BASE_FEATURES + Z_FEATURES,
    "base_frag": BASE_FEATURES + FRAG_FEATURES,
    "base_z_frag": BASE_FEATURES + Z_FEATURES + FRAG_FEATURES,
    "base_z_frag_v2": BASE_FEATURES + Z_FEATURES + FRAG_FEATURES + FRAG_V2_FEATURES,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return feval.load_rows(Path(path))


def truth(row: dict[str, Any], truth_column: str = "candidate_is_true") -> bool:
    value = row.get(truth_column)
    if value is None and truth_column != "candidate_is_true":
        value = row.get("candidate_is_true")
    return str(value).strip().lower() in {"1", "true", "yes"} or row.get("is_true") is True


def infer_features(
    rows: list[dict[str, Any]],
    extra_exclude: Iterable[str] = (),
    sample_rows: int = 2000,
) -> list[str]:
    if not rows:
        return []
    exclude = set(DEFAULT_EXCLUDE) | set(extra_exclude)
    features: list[str] = []
    sample = rows[:sample_rows]
    for key in rows[0].keys():
        if key in exclude:
            continue
        ok = False
        for row in sample:
            value = feval.fnum(row.get(key), math.nan)
            if math.isfinite(value):
                ok = True
                break
        if ok:
            features.append(key)
    for feat in ["log_freq", "inv_freq_rank", "has_bfn_stats_num"]:
        if feat not in features:
            features.append(feat)
    return features


def parse_features(features: str | None, rows: list[dict[str, Any]]) -> list[str]:
    if features:
        return [x.strip() for x in features.split(",") if x.strip()]
    return infer_features(rows)


def preset_features(name: str, rows: list[dict[str, Any]], include_group_relative: bool = False) -> list[str]:
    if name not in FEATURE_PRESETS:
        raise ValueError(f"Unknown feature preset: {name}. Choices: {sorted(FEATURE_PRESETS)}")
    available = set(rows[0].keys()) if rows else set()
    features = [feat for feat in FEATURE_PRESETS[name] if feat in available or feat in {"log_freq", "inv_freq_rank", "freq_rank", "has_bfn_stats_num"}]
    if include_group_relative and rows:
        bases = list(features)
        for feat in bases:
            for suffix in ("_group_z", "_group_rank", "_group_gap_top1", "_group_percentile", "_group_minmax"):
                col = f"{feat}{suffix}"
                if col in available and col not in features:
                    features.append(col)
    return features


def select_features(
    rows: list[dict[str, Any]],
    features: str | None = None,
    preset: str | None = None,
    include_group_relative: bool = False,
) -> list[str]:
    if features and preset:
        raise ValueError("Use either explicit --features or --feature-preset, not both.")
    if features:
        return [x.strip() for x in features.split(",") if x.strip()]
    if preset:
        return preset_features(preset, rows, include_group_relative)
    return infer_features(rows)


def group_rows_ordered(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        spec_id = str(row.get("spec_id"))
        if spec_id not in groups:
            groups[spec_id] = []
            order.append(spec_id)
        groups[spec_id].append(row)
    return [(spec_id, groups[spec_id]) for spec_id in order]


def is_group_contiguous(rows: list[dict[str, Any]]) -> bool:
    seen: set[str] = set()
    current = None
    for row in rows:
        spec_id = str(row.get("spec_id"))
        if spec_id != current:
            if spec_id in seen:
                return False
            if current is not None:
                seen.add(current)
            current = spec_id
    return True


def sort_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("spec_id")),
            int(feval.fnum(row.get("candidate_rank_freq"), 10**9)),
            str(row.get("candidate_hash")),
        ),
    )


def prepare_grouped_rows(rows: list[dict[str, Any]], mode: str = "sort") -> list[dict[str, Any]]:
    if mode == "none":
        return rows
    if is_group_contiguous(rows):
        return rows
    if mode == "check":
        raise ValueError("spec_id rows are not contiguous. Sort by spec_id/candidate_rank_freq before listwise ranking.")
    if mode == "sort":
        return sort_by_group(rows)
    raise ValueError(f"Unknown group order mode: {mode}")


def filter_groups(
    rows: list[dict[str, Any]],
    positive_only: bool = True,
    max_groups: int | None = None,
    truth_column: str = "candidate_is_true",
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    n_groups = 0
    for _, candidates in group_rows_ordered(rows):
        if positive_only and not any(truth(row, truth_column) for row in candidates):
            continue
        kept.extend(candidates)
        n_groups += 1
        if max_groups is not None and n_groups >= max_groups:
            break
    return kept


def matrix(rows: list[dict[str, Any]], features: list[str]):
    import numpy as np

    return np.asarray(
        [[feval.fnum(row.get(feat), 0.0) for feat in features] for row in rows],
        dtype="float32",
    )


def labels(rows: list[dict[str, Any]], truth_column: str = "candidate_is_true"):
    import numpy as np

    return np.asarray([1 if truth(row, truth_column) else 0 for row in rows], dtype="float32")


def group_sizes(rows: list[dict[str, Any]]) -> list[int]:
    return [len(candidates) for _, candidates in group_rows_ordered(rows)]


def group_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("spec_id")) for row in rows]


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("spec_id")), str(row.get("candidate_hash")))


def write_scores(path: str | Path, rows: list[dict[str, Any]], scores: Iterable[float], column: str = "score") -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spec_id", "candidate_hash", column], delimiter="\t")
        writer.writeheader()
        for row, score in zip(rows, scores):
            writer.writerow(
                {
                    "spec_id": row.get("spec_id"),
                    "candidate_hash": row.get("candidate_hash"),
                    column: float(score),
                }
            )


def score_map(rows: list[dict[str, Any]], scores: Iterable[float]) -> dict[tuple[str, str], float]:
    return {row_key(row): float(score) for row, score in zip(rows, scores)}


def evaluate_from_scores(
    rows: list[dict[str, Any]],
    scores: Iterable[float],
    name: str,
    topk: list[int] | None = None,
) -> dict[str, Any]:
    topk = topk or [1, 5, 10]
    mapping = score_map(rows, scores)
    return feval.evaluate_scores(rows, lambda row: mapping.get(row_key(row), 0.0), topk, name)


def group_zscore(rows: list[dict[str, Any]], values: Iterable[float]) -> list[float]:
    vals = [float(v) for v in values]
    out = [0.0] * len(vals)
    start = 0
    for _, candidates in group_rows_ordered(rows):
        end = start + len(candidates)
        part = vals[start:end]
        if part:
            mean = sum(part) / len(part)
            var = sum((v - mean) ** 2 for v in part) / len(part)
            sd = math.sqrt(var) if var > 1e-12 else 1.0
            for idx, value in enumerate(part, start=start):
                out[idx] = (value - mean) / sd
        start = end
    return out


def fuse_with_prior(
    rows: list[dict[str, Any]],
    model_scores: Iterable[float],
    prior_score_column: str | None,
    prior_weight: float,
    model_weight: float,
    group_normalize: bool,
) -> list[float]:
    model_vals = [float(v) for v in model_scores]
    if not prior_score_column:
        return model_vals
    prior_vals = [feval.fnum(row.get(prior_score_column), 0.0) for row in rows]
    if group_normalize:
        model_vals = group_zscore(rows, model_vals)
        prior_vals = group_zscore(rows, prior_vals)
    return [prior_weight * p + model_weight * m for p, m in zip(prior_vals, model_vals)]


def feature_stats(rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, float]]:
    import numpy as np

    x = matrix(rows, features)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        feat: {"mean": float(mu), "std": float(sd)}
        for feat, mu, sd in zip(features, mean.tolist(), std.tolist())
    }


def apply_feature_stats(rows: list[dict[str, Any]], features: list[str], stats: dict[str, dict[str, float]]):
    import numpy as np

    x = matrix(rows, features)
    mean = np.asarray([stats[feat]["mean"] for feat in features], dtype="float32")
    std = np.asarray([stats[feat]["std"] for feat in features], dtype="float32")
    return (x - mean) / std


def dump_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
