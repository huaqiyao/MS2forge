#!/usr/bin/env python3
"""Evaluate MS2Forge rerank scores on a candidate table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ID_COLUMNS = {
    "spec_id",
    "smiles",
    "candidate_hash",
    "candidate_is_true",
    "true_rank",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-table", required=True)
    p.add_argument("--score-column", default="log_freq")
    p.add_argument("--score-file", default=None, help="Optional TSV with spec_id, candidate_hash, score")
    p.add_argument("--score-file-column", default="score")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--topk", default="1,5,10")
    p.add_argument("--write-ranked", action="store_true")
    return p.parse_args()


def fnum(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        low = value.lower()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    add_derived_features(rows)
    return rows


def add_derived_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        freq = fnum(row.get("candidate_freq"), 0.0)
        rank = fnum(row.get("candidate_rank_freq"), 1e9)
        row["log_freq"] = math.log(freq + 1.0)
        row["inv_freq_rank"] = 1.0 / max(rank, 1.0)
        row["is_true"] = str(row.get("candidate_is_true", "")).strip().lower() in {"1", "true", "yes"}
        row["freq_rank"] = int(rank)
        row["has_bfn_stats_num"] = fnum(row.get("has_bfn_stats"), 0.0)


def load_score_file(path: Path, score_column: str) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (str(row.get("spec_id")), str(row.get("candidate_hash")))
            scores[key] = fnum(row.get(score_column), 0.0)
    return scores


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("spec_id"))].append(row)
    return groups


def true_rank_from_ranked(ranked: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(ranked, start=1):
        if row.get("is_true"):
            return idx
    return None


def evaluate_scores(
    rows: list[dict[str, Any]],
    score_getter: Callable[[dict[str, Any]], float],
    topk: list[int],
    name: str,
) -> dict[str, Any]:
    groups = group_rows(rows)
    topk_hits = {k: 0 for k in topk}
    present_topk_hits = {k: 0 for k in topk}
    n_specs = 0
    present = 0
    baseline_top1 = 0
    score_top1 = 0
    baseline_present_top1 = 0
    score_present_top1 = 0
    promotion = 0
    demotion = 0
    rank2_10_rescue = 0
    mrr = 0.0
    present_mrr = 0.0

    for candidates in groups.values():
        n_specs += 1
        baseline_ranked = sorted(candidates, key=lambda r: (int(fnum(r.get("candidate_rank_freq"), 1e9)),))
        scored_ranked = sorted(candidates, key=lambda r: (-score_getter(r), int(fnum(r.get("candidate_rank_freq"), 1e9))))
        baseline_true_rank = true_rank_from_ranked(baseline_ranked)
        score_true_rank = true_rank_from_ranked(scored_ranked)

        if baseline_true_rank is not None:
            present += 1
            if baseline_true_rank == 1:
                baseline_present_top1 += 1
            if score_true_rank == 1:
                score_present_top1 += 1

        baseline_correct = baseline_true_rank == 1
        score_correct = score_true_rank == 1
        baseline_top1 += int(baseline_correct)
        score_top1 += int(score_correct)
        if (not baseline_correct) and score_correct:
            promotion += 1
            if baseline_true_rank is not None and 2 <= baseline_true_rank <= 10:
                rank2_10_rescue += 1
        if baseline_correct and not score_correct:
            demotion += 1
        if score_true_rank is not None:
            mrr += 1.0 / score_true_rank
            present_mrr += 1.0 / score_true_rank
            for k in topk:
                if score_true_rank <= k:
                    topk_hits[k] += 1
                    present_topk_hits[k] += 1

    result: dict[str, Any] = {
        "name": name,
        "n_specs": n_specs,
        "present": present,
        "present_rate": present / n_specs if n_specs else 0.0,
        "baseline_top1": baseline_top1,
        "baseline_top1_rate": baseline_top1 / n_specs if n_specs else 0.0,
        "top1": score_top1,
        "top1_rate": score_top1 / n_specs if n_specs else 0.0,
        "baseline_present_top1": baseline_present_top1,
        "baseline_present_top1_rate": baseline_present_top1 / present if present else 0.0,
        "present_top1": score_present_top1,
        "present_top1_rate": score_present_top1 / present if present else 0.0,
        "mrr": mrr / n_specs if n_specs else 0.0,
        "present_mrr": present_mrr / present if present else 0.0,
        "promotion": promotion,
        "demotion": demotion,
        "net_gain": promotion - demotion,
        "rank2_10_rescue": rank2_10_rescue,
    }
    for k in topk:
        result[f"top{k}"] = topk_hits[k]
        result[f"top{k}_rate"] = topk_hits[k] / n_specs if n_specs else 0.0
        result[f"present_top{k}"] = present_topk_hits[k]
        result[f"present_top{k}_rate"] = present_topk_hits[k] / present if present else 0.0
    return result


def ranked_rows(rows: list[dict[str, Any]], score_getter: Callable[[dict[str, Any]], float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec_id, candidates in group_rows(rows).items():
        ranked = sorted(candidates, key=lambda r: (-score_getter(r), int(fnum(r.get("candidate_rank_freq"), 1e9))))
        for rank, row in enumerate(ranked, start=1):
            copied = dict(row)
            copied["rerank_rank"] = rank
            copied["rerank_score"] = score_getter(row)
            out.append(copied)
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.candidate_table))
    topk = [int(x) for x in args.topk.split(",") if x.strip()]

    if args.score_file:
        score_map = load_score_file(Path(args.score_file), args.score_file_column)

        def score_getter(row: dict[str, Any]) -> float:
            return score_map.get((str(row.get("spec_id")), str(row.get("candidate_hash"))), 0.0)

        name = args.name or Path(args.score_file).stem
    else:
        score_col = args.score_column

        def score_getter(row: dict[str, Any]) -> float:
            return fnum(row.get(score_col), 0.0)

        name = args.name or score_col

    metrics = evaluate_scores(rows, score_getter, topk, name)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.write_ranked:
        write_tsv(out_dir / "ranked_candidates.tsv", ranked_rows(rows, score_getter))


if __name__ == "__main__":
    main()
