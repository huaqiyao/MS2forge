#!/usr/bin/env python3
"""Train a graded similarity ranker and fuse it with frozen EvidenceRank.

Model selection is validation-only. The frozen exact head is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rerank import training_features as fut
from rerank import training_common as tc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", required=True, type=Path)
    parser.add_argument("--valid-table", required=True, type=Path)
    parser.add_argument("--train-targets", required=True, type=Path)
    parser.add_argument("--valid-targets", required=True, type=Path)
    parser.add_argument("--features-from-metrics", required=True, type=Path)
    parser.add_argument("--exact-valid-scores", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-column", default="candidate_tanimoto_to_truth")
    parser.add_argument("--truth-column", default="candidate_is_true")
    parser.add_argument("--relevance-bins", type=int, default=15)
    parser.add_argument("--similarity-exponent", type=float, default=1.0)
    parser.add_argument("--exact-boost", type=int, default=0)
    parser.add_argument(
        "--truth-absent-group-weight",
        type=float,
        default=1.0,
        help="Train-row weight for candidate groups in which the exact truth is absent.",
    )
    parser.add_argument("--fusion-exact-weights", default="0,0.25,0.5,0.65,0.75,0.85,0.9,0.95,1")
    parser.add_argument("--max-top1-drop", type=int, default=0)
    parser.add_argument("--max-top10-drop", type=int, default=0)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="Smoke-test only: read at most this many rows from each candidate table.",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--colsample", type=float, default=0.85)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.set_defaults(**tc.recipe("stage_a")["parameters"])
    tc.add_data_arguments(parser)
    return parser.parse_args()


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return fut.load_rows(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    fut.feval.add_derived_features(rows)
    return rows


def attach_targets(rows: list[dict[str, Any]], path: Path, target_column: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"spec_id", "candidate_hash", target_column}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Missing target columns in {path}: {missing}")
        count = 0
        for count, (row, target) in enumerate(zip(rows, reader), start=1):
            row_key = (str(row.get("spec_id")), str(row.get("candidate_hash")))
            target_key = (str(target.get("spec_id")), str(target.get("candidate_hash")))
            if row_key != target_key:
                raise SystemExit(
                    f"Target alignment mismatch at row {count}: table={row_key}, sidecar={target_key}"
                )
            value = float(target[target_column])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid similarity target at row {count}")
            row[target_column] = value
        extra = next(reader, None)
        if count != len(rows) or extra is not None:
            raise SystemExit(
                f"Target row-count mismatch for {path}: table={len(rows)}, aligned={count}, extra={extra is not None}"
            )


def load_score_map(path: Path) -> dict[tuple[str, str], float]:
    mapping: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = str(row["spec_id"]), str(row["candidate_hash"])
            value = float(row["score"])
            if key in mapping or not math.isfinite(value):
                raise ValueError(f"Duplicate key or invalid EvidenceRank score: {key}")
            mapping[key] = value
    return mapping


def group_normalize(rows: list[dict[str, Any]], values: Iterable[float]) -> np.ndarray:
    output = np.asarray(list(values), dtype=np.float64)
    normalized = np.zeros_like(output)
    start = 0
    for _, candidates in fut.group_rows_ordered(rows):
        stop = start + len(candidates)
        part = output[start:stop]
        sd = float(part.std())
        if sd < 1e-12:
            normalized[start:stop] = 0.0
        else:
            normalized[start:stop] = (part - float(part.mean())) / sd
        start = stop
    return normalized


def ranked_indices(candidates: list[dict[str, Any]], scores: np.ndarray, offset: int) -> list[int]:
    local = list(range(len(candidates)))

    def key(index: int) -> tuple[float, int, str]:
        row = candidates[index]
        try:
            frequency_rank = int(float(row.get("candidate_rank_freq", 10**9)))
        except (TypeError, ValueError):
            frequency_rank = 10**9
        return (-float(scores[offset + index]), frequency_rank, str(row.get("candidate_hash")))

    local.sort(key=key)
    return local


def evaluate(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    target_column: str,
    truth_column: str,
) -> dict[str, Any]:
    n_specs = 0
    exact_at_1 = 0
    exact_at_10 = 0
    tanimoto_at_1 = 0.0
    tanimoto_at_10 = 0.0
    truth_present = 0
    offset = 0
    for _, candidates in fut.group_rows_ordered(rows):
        order = ranked_indices(candidates, scores, offset)
        top1 = candidates[order[0]]
        top10 = [candidates[index] for index in order[:10]]
        exact_at_1 += int(truth(top1.get(truth_column)))
        exact_at_10 += int(any(truth(row.get(truth_column)) for row in top10))
        tanimoto_at_1 += float(top1[target_column])
        tanimoto_at_10 += max(float(row[target_column]) for row in top10)
        truth_present += int(any(truth(row.get(truth_column)) for row in candidates))
        n_specs += 1
        offset += len(candidates)
    return {
        "n_specs": n_specs,
        "truth_present": truth_present,
        "truth_present_rate": truth_present / n_specs if n_specs else 0.0,
        "exact_at_1": exact_at_1,
        "exact_at_1_rate": exact_at_1 / n_specs if n_specs else 0.0,
        "exact_at_10": exact_at_10,
        "exact_at_10_rate": exact_at_10 / n_specs if n_specs else 0.0,
        "tanimoto_at_1": tanimoto_at_1 / n_specs if n_specs else 0.0,
        "tanimoto_at_10": tanimoto_at_10 / n_specs if n_specs else 0.0,
    }


def relevance_labels(
    rows: list[dict[str, Any]],
    target_column: str,
    truth_column: str,
    bins: int,
    exponent: float,
    exact_boost: int,
) -> np.ndarray:
    labels = []
    for row in rows:
        similarity = min(1.0, max(0.0, float(row[target_column])))
        grade = int(round(bins * (similarity**exponent)))
        if truth(row.get(truth_column)):
            grade += exact_boost
        labels.append(grade)
    return np.asarray(labels, dtype=np.int32)


def group_domain_weights(
    rows: list[dict[str, Any]],
    truth_column: str,
    truth_absent_group_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.ones(len(rows), dtype=np.float64)
    offset = 0
    present_groups = 0
    absent_groups = 0
    for _, candidates in fut.group_rows_ordered(rows):
        stop = offset + len(candidates)
        if any(truth(row.get(truth_column)) for row in candidates):
            present_groups += 1
        else:
            absent_groups += 1
            weights[offset:stop] = truth_absent_group_weight
        offset = stop
    return weights, {
        "truth_present_groups": present_groups,
        "truth_absent_groups": absent_groups,
        "truth_absent_group_weight": truth_absent_group_weight,
    }


def write_scores(path: Path, rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spec_id", "candidate_hash", "score"], delimiter="\t")
        writer.writeheader()
        for row, score in zip(rows, scores):
            writer.writerow(
                {
                    "spec_id": row.get("spec_id"),
                    "candidate_hash": row.get("candidate_hash"),
                    "score": f"{float(score):.10g}",
                }
            )


def main() -> None:
    args = parse_args()
    tc.validate_training_inputs(args)
    if (
        args.relevance_bins < 1
        or args.exact_boost < 0
        or args.similarity_exponent <= 0
        or args.truth_absent_group_weight <= 0
    ):
        raise SystemExit(
            "relevance-bins >= 1, exact-boost >= 0, similarity-exponent > 0, "
            "and truth-absent-group-weight > 0 are required"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fut.set_seed(args.seed)

    feature_metadata = json.loads(args.features_from_metrics.read_text(encoding="utf-8"))
    features = list(feature_metadata["features"])
    train_rows = load_rows(args.train_table, args.max_input_rows)
    valid_rows = load_rows(args.valid_table, args.max_input_rows)
    attach_targets(train_rows, args.train_targets, args.target_column)
    attach_targets(valid_rows, args.valid_targets, args.target_column)
    tc.validate_features(train_rows, features)
    tc.validate_features(valid_rows, features)
    train_rows = fut.prepare_grouped_rows(train_rows, "sort")
    valid_rows = fut.prepare_grouped_rows(valid_rows, "sort")
    if args.max_train_groups is not None:
        train_rows = fut.filter_groups(
            train_rows,
            positive_only=False,
            max_groups=args.max_train_groups,
            truth_column=args.truth_column,
        )
    missing_features = sorted(set(features) - set(train_rows[0]))
    if missing_features:
        raise SystemExit(f"Missing {len(missing_features)} model features: {missing_features[:20]}")

    labels = relevance_labels(
        train_rows,
        args.target_column,
        args.truth_column,
        args.relevance_bins,
        args.similarity_exponent,
        args.exact_boost,
    )
    train_row_weights, domain_weighting = group_domain_weights(
        train_rows,
        args.truth_column,
        args.truth_absent_group_weight,
    )
    max_label = int(labels.max(initial=0))
    from lightgbm import LGBMRanker

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(max_label + 1)),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        colsample_bytree=args.colsample,
        reg_lambda=args.l2,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    train_matrix = fut.matrix(train_rows, features)
    model.fit(
        train_matrix,
        labels,
        group=fut.group_sizes(train_rows),
        sample_weight=train_row_weights,
    )
    del train_matrix

    similarity_scores = np.asarray(model.predict(fut.matrix(valid_rows, features)), dtype=np.float64)
    exact_map = load_score_map(args.exact_valid_scores)
    exact_scores = np.asarray(
        [exact_map[fut.row_key(row)] for row in valid_rows], dtype=np.float64
    )
    exact_normalized = group_normalize(valid_rows, exact_scores)
    similarity_normalized = group_normalize(valid_rows, similarity_scores)
    baseline_metrics = evaluate(valid_rows, exact_normalized, args.target_column, args.truth_column)

    weights = sorted({float(value) for value in args.fusion_exact_weights.split(",")})
    if not weights or min(weights) < 0.0 or max(weights) > 1.0:
        raise SystemExit("fusion-exact-weights must contain values in [0, 1]")
    grid: list[dict[str, Any]] = []
    selected_scores = None
    selected_record = None
    for exact_weight in weights:
        fused = exact_weight * exact_normalized + (1.0 - exact_weight) * similarity_normalized
        metrics = evaluate(valid_rows, fused, args.target_column, args.truth_column)
        passes = (
            metrics["exact_at_1"] >= baseline_metrics["exact_at_1"] - args.max_top1_drop
            and metrics["exact_at_10"] >= baseline_metrics["exact_at_10"] - args.max_top10_drop
        )
        record = {"exact_weight": exact_weight, "similarity_weight": 1.0 - exact_weight, "passes_exact_guard": passes, **metrics}
        grid.append(record)
        key = (
            int(passes),
            metrics["tanimoto_at_1"] if passes else -math.inf,
            metrics["tanimoto_at_10"] if passes else -math.inf,
            metrics["exact_at_1"],
            metrics["exact_at_10"],
            exact_weight,
        )
        if selected_record is None or key > selected_record[0]:
            selected_record = (key, record)
            selected_scores = fused

    assert selected_record is not None and selected_scores is not None
    selected = selected_record[1]
    if not selected["passes_exact_guard"]:
        raise ValueError("No fusion weight passed validation; no Stage A model published")
    write_scores(args.out_dir / "valid_similarity_head_scores.tsv", valid_rows, similarity_scores)
    write_scores(args.out_dir / "valid_selected_fused_scores.tsv", valid_rows, selected_scores)
    with (args.out_dir / "valid_fusion_grid.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(grid)

    importance = sorted(
        zip(features, [float(value) for value in model.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )
    metadata = {
        "schema": "ms2forge.er_pareto.v1",
        "status": "experimental_not_frozen",
        "selection_split": "validation_only",
        "test_access": "closed",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_train_rows": len(train_rows),
        "n_valid_rows": len(valid_rows),
        "n_features": len(features),
        "features": features,
        "relevance": {
            "formula": "round(relevance_bins * tanimoto ** similarity_exponent) + exact_boost * is_exact",
            "max_label": max_label,
            "all_candidate_groups_used": True,
        },
        "domain_weighting": domain_weighting,
        "frozen_evidencerank_baseline": baseline_metrics,
        "fusion_grid": grid,
        "selected": selected,
        "selection_key": [
            "passes exact Top-1/Top-10 non-inferiority guard",
            "max validation Tanimoto@1",
            "max validation Tanimoto@10",
            "max exact@1",
            "max exact@10",
            "higher frozen-exact weight",
        ],
        "feature_importance": importance,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "stage_a_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "features": features,
                "metadata": metadata,
                "selected_exact_weight": selected["exact_weight"],
            },
            handle,
        )
    print(json.dumps({"baseline": baseline_metrics, "selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
