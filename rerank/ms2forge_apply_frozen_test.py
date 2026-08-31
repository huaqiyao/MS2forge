#!/usr/bin/env python3
"""Apply the validation-frozen Stage-A and SSR models to the MSG test set.

Scoring commands in this file never read truth structures, exact-match labels,
or test similarity targets.  The final score file is hashed and frozen before
the separate FLASH-UEP evaluator is allowed to access the test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ms2forge_spectrum_structure_ranker as ssr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_hash(path: Path, expected: str | None, label: str) -> str:
    observed = sha256_file(path)
    if expected and observed != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected}, observed={observed}")
    return observed


def ensure_contiguous_or_sort(frame: pd.DataFrame) -> pd.DataFrame:
    spec = frame["spec_id"].astype(str)
    group_starts = spec.ne(spec.shift(fill_value=spec.iloc[0] if len(spec) else ""))
    sequence = spec[group_starts]
    if sequence.duplicated().any():
        rank = pd.to_numeric(frame["candidate_rank_freq"], errors="coerce").fillna(10**9)
        frame = frame.assign(_rank_for_sort=rank, _row_for_sort=np.arange(len(frame)))
        frame = frame.sort_values(
            ["spec_id", "_rank_for_sort", "candidate_hash", "_row_for_sort"],
            kind="mergesort",
        ).drop(columns=["_rank_for_sort", "_row_for_sort"])
    return frame.reset_index(drop=True)


def group_zscore_population(spec_ids: pd.Series, values: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(spec_ids.astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sums = np.bincount(codes, weights=values)
    sums2 = np.bincount(codes, weights=values * values)
    means = sums / counts
    variances = np.maximum(sums2 / counts - means * means, 0.0)
    stds = np.sqrt(variances)
    denom = np.where(stds > 1e-12, stds, 1.0)
    return (values - means[codes]) / denom[codes]


def group_zscore_sample(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("spec_id", sort=False)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").fillna(0.0)
    return (frame[column] - mean) / std.where(std > 1e-12, 1.0)


def compare_expected(actual: pd.DataFrame, expected_path: Path, tolerance: float) -> dict:
    expected = pd.read_csv(
        expected_path,
        sep="\t",
        dtype={"spec_id": str, "candidate_hash": str},
    ).rename(columns={"score": "expected_score"})
    joined = actual.merge(
        expected,
        on=["spec_id", "candidate_hash"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"expected-score key mismatch: {joined['_merge'].value_counts().to_dict()}")
    delta = np.abs(joined["score"].to_numpy(float) - joined["expected_score"].to_numpy(float))
    maximum = float(delta.max(initial=0.0))
    if maximum > tolerance:
        raise RuntimeError(f"inference self-check failed: max_abs_delta={maximum} > {tolerance}")
    return {"expected_scores_sha256": sha256_file(expected_path), "max_abs_delta": maximum}


def command_stage_a(args: argparse.Namespace) -> None:
    model_path = Path(args.model_pkl).resolve()
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["model"]
    features = list(bundle["features"])
    frozen_weight = float(bundle["selected_exact_weight"])
    if abs(frozen_weight - args.exact_weight) > 1e-12:
        raise RuntimeError(f"Stage-A weight mismatch: checkpoint={frozen_weight}, requested={args.exact_weight}")

    required = list(dict.fromkeys(["spec_id", "candidate_hash", "candidate_rank_freq", *features]))
    frame = pd.read_csv(
        args.candidate_table,
        sep="\t",
        usecols=required,
        dtype={"spec_id": str, "candidate_hash": str},
        keep_default_na=False,
        low_memory=False,
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"candidate table lacks frozen features: {missing[:20]}")
    if frame.duplicated(["spec_id", "candidate_hash"]).any():
        raise RuntimeError("duplicate candidate keys in Stage-A input")
    frame = ensure_contiguous_or_sort(frame)

    matrix = np.column_stack(
        [pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(np.float32) for name in features]
    )
    similarity = np.asarray(model.predict(matrix), dtype=np.float64)
    exact = pd.read_csv(
        args.evidencerank_scores,
        sep="\t",
        dtype={"spec_id": str, "candidate_hash": str},
    ).rename(columns={"score": "exact_score"})
    frame = frame[["spec_id", "candidate_hash", "candidate_rank_freq"]].assign(_row=np.arange(len(frame)))
    frame = frame.merge(exact, on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    frame = frame.sort_values("_row", kind="mergesort").reset_index(drop=True)
    if frame["exact_score"].isna().any() or len(exact) != len(frame):
        raise RuntimeError("EvidenceRank score join is incomplete")

    exact_z = group_zscore_population(frame["spec_id"], frame["exact_score"].to_numpy(np.float64))
    similarity_z = group_zscore_population(frame["spec_id"], similarity)
    scores = args.exact_weight * exact_z + (1.0 - args.exact_weight) * similarity_z
    output = frame[["spec_id", "candidate_hash"]].assign(score=scores)
    self_check = None
    if args.expected_scores:
        self_check = compare_expected(output, Path(args.expected_scores), args.tolerance)

    out_path = Path(args.out_scores).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, sep="\t", index=False, float_format="%.10g")
    summary = {
        "schema": "ms2forge.stage_a_frozen_inference.v1",
        "status": "frozen_blind_inference",
        "scored_split": args.declared_split,
        "truth_columns_read": [],
        "n_candidates": int(len(output)),
        "n_spectra": int(output["spec_id"].nunique()),
        "fusion": {"evidencerank_weight": args.exact_weight, "similarity_weight": 1.0 - args.exact_weight},
        "input_sha256": {
            "model": ensure_hash(model_path, args.expected_model_sha256, "Stage-A model"),
            "evidencerank_scores": ensure_hash(Path(args.evidencerank_scores), args.expected_evidencerank_sha256, "EvidenceRank scores"),
        },
        "output_sha256": sha256_file(out_path),
        "self_check": self_check,
    }
    dump_json(summary, out_path.with_suffix(".json"))
    print(json.dumps(summary, indent=2), flush=True)


def command_prepare_test(args: argparse.Namespace) -> None:
    if args.declared_split.lower() != "test":
        raise ValueError("test preparation requires --declared-split test")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["spec_id", "candidate_hash", "candidate_canonical_smiles", "candidate_rank_freq"]
    candidates = pd.read_csv(
        args.candidate_table,
        sep="\t",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
    )
    if candidates.duplicated(["spec_id", "candidate_hash"]).any():
        raise RuntimeError("duplicate candidate keys in test candidate table")
    spec_ids = candidates["spec_id"].drop_duplicates().tolist()
    split_counts = ssr.verify_split(spec_ids, Path(args.split_table), {"test"})
    # Deliberately exclude the truth-SMILES column from the labels file.
    labels = pd.read_csv(
        args.labels,
        sep="\t",
        usecols=["spec", "ionization", "formula", "instrument"],
        dtype=str,
        keep_default_na=False,
    )
    spectra_audit = ssr.build_spectrum_arrays(
        spec_ids, labels, Path(args.spec_dir), args.max_peaks, out_dir
    )
    spec_to_index = {spec_id: index for index, spec_id in enumerate(spec_ids)}
    np.save(out_dir / "candidate_spec_index.npy", candidates["spec_id"].map(spec_to_index).to_numpy(np.int32))
    packed = np.lib.format.open_memmap(
        out_dir / "candidate_fp_packed.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(candidates), ssr.FP_BYTES),
    )
    invalid = 0
    for index, (fingerprint, ok) in enumerate(
        ssr.parallel_map(ssr._fingerprint_single, candidates["candidate_canonical_smiles"].tolist(), args.workers)
    ):
        packed[index] = fingerprint
        invalid += int(not ok)
        if index and index % 25000 == 0:
            print(json.dumps({"fingerprinted_candidates": index, "total": len(candidates)}), flush=True)
    packed.flush()
    candidates[["spec_id", "candidate_hash", "candidate_rank_freq"]].to_csv(
        out_dir / "candidate_rows.tsv", sep="\t", index=False
    )
    pd.DataFrame({"spec_id": spec_ids}).to_csv(out_dir / "spectra.tsv", sep="\t", index=False)
    summary = {
        "schema": "ms2forge.ssr_test_cache.v1",
        "status": "frozen_blind_inference",
        "source_split": "test_only",
        "truth_columns_read": [],
        "metadata_columns_read": ["spec", "ionization", "formula", "instrument"],
        "n_candidates": int(len(candidates)),
        "n_spectra": int(len(spec_ids)),
        "invalid_candidate_fingerprints": invalid,
        "split_counts": split_counts,
        "spectra_audit": spectra_audit,
        "max_peaks": args.max_peaks,
        "candidate_table_sha256": sha256_file(Path(args.candidate_table)),
    }
    dump_json(summary, out_dir / "prepare_summary.json")
    (out_dir / "TEST_CACHE_READY_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def command_score_ssr(args: argparse.Namespace) -> None:
    torch, _, _ = ssr.import_torch()
    cache = Path(args.candidate_cache).resolve()
    prepare = json.loads((cache / "prepare_summary.json").read_text(encoding="utf-8"))
    if prepare.get("source_split") != "test_only" or prepare.get("truth_columns_read") != []:
        raise RuntimeError("SSR scoring requires an audited blind-test cache")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = ensure_hash(checkpoint_path, args.expected_checkpoint_sha256, "SSR checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ssr.make_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    peaks = np.load(cache / "spectra_peaks.npy", mmap_mode="r")
    meta = np.load(cache / "spectra_meta.npy", mmap_mode="r")
    packed = np.load(cache / "candidate_fp_packed.npy", mmap_mode="r")
    spec_index = np.load(cache / "candidate_spec_index.npy", mmap_mode="r")
    rows = pd.read_csv(cache / "candidate_rows.tsv", sep="\t", dtype=str, keep_default_na=False)
    spectrum_embeddings = []
    with torch.no_grad():
        for start in range(0, len(peaks), args.batch_size):
            stop = min(start + args.batch_size, len(peaks))
            p = torch.from_numpy(np.array(peaks[start:stop], dtype=np.float32, copy=True)).to(device)
            m = torch.from_numpy(np.array(meta[start:stop], dtype=np.float32, copy=True)).to(device)
            spectrum_embeddings.append(model.encode_spectrum(p, m).cpu().numpy().astype(np.float32))
    spectrum_embeddings = np.concatenate(spectrum_embeddings, axis=0)
    scores = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            fp = torch.from_numpy(ssr.unpack_fingerprints(np.asarray(packed[start:stop]))).to(device)
            molecule = model.encode_molecule(fp).cpu().numpy()
            spectrum = spectrum_embeddings[np.asarray(spec_index[start:stop], dtype=np.int64)]
            scores[start:stop] = np.sum(spectrum * molecule, axis=1)
    out_path = Path(args.out_scores).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"spec_id": rows["spec_id"], "candidate_hash": rows["candidate_hash"], "score": scores}).to_csv(
        out_path, sep="\t", index=False, float_format="%.9g"
    )
    summary = {
        "schema": "ms2forge.ssr_frozen_test_scores.v1",
        "status": "frozen_blind_inference",
        "scored_split": "test_only",
        "truth_columns_read": [],
        "device": str(device),
        "n_rows": int(len(rows)),
        "checkpoint_sha256": checkpoint_hash,
        "scores_sha256": sha256_file(out_path),
    }
    dump_json(summary, out_path.with_suffix(".json"))
    print(json.dumps(summary, indent=2), flush=True)


def command_fuse(args: argparse.Namespace) -> None:
    rows = pd.read_csv(
        Path(args.candidate_cache) / "candidate_rows.tsv",
        sep="\t",
        dtype={"spec_id": str, "candidate_hash": str},
        keep_default_na=False,
    )
    base = pd.read_csv(args.stage_a_scores, sep="\t", dtype={"spec_id": str, "candidate_hash": str}).rename(columns={"score": "base_score"})
    neural = pd.read_csv(args.neural_scores, sep="\t", dtype={"spec_id": str, "candidate_hash": str}).rename(columns={"score": "neural_score"})
    frame = rows.merge(base, on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    frame = frame.merge(neural, on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    if len(frame) != len(base) or len(frame) != len(neural) or frame[["base_score", "neural_score"]].isna().any().any():
        raise RuntimeError("fixed-fusion score join is incomplete")
    frame["base_z"] = group_zscore_sample(frame, "base_score")
    frame["neural_z"] = group_zscore_sample(frame, "neural_score")
    frame["score"] = args.base_weight * frame["base_z"] + (1.0 - args.base_weight) * frame["neural_z"]
    output = frame[["spec_id", "candidate_hash", "score"]]
    self_check = None
    if args.expected_scores:
        self_check = compare_expected(output, Path(args.expected_scores), args.tolerance)
    out_path = Path(args.out_scores).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, sep="\t", index=False, float_format="%.9g")
    summary = {
        "schema": "ms2forge.postfreeze_frozen_test_predictions.v1",
        "status": "predictions_frozen_before_truth_evaluation",
        "scored_split": args.declared_split,
        "truth_columns_read": [],
        "n_candidates": int(len(output)),
        "n_spectra": int(output["spec_id"].nunique()),
        "fusion": {"stage_a_weight": args.base_weight, "ssr_weight": 1.0 - args.base_weight},
        "input_sha256": {
            "stage_a_scores": sha256_file(Path(args.stage_a_scores)),
            "neural_scores": sha256_file(Path(args.neural_scores)),
        },
        "output_sha256": sha256_file(out_path),
        "self_check": self_check,
    }
    dump_json(summary, out_path.with_suffix(".json"))
    (out_path.parent / "PREDICTIONS_FROZEN_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stage-a")
    p.add_argument("--candidate-table", required=True, type=Path)
    p.add_argument("--model-pkl", required=True, type=Path)
    p.add_argument("--evidencerank-scores", required=True, type=Path)
    p.add_argument("--out-scores", required=True, type=Path)
    p.add_argument("--declared-split", choices=["validation", "test"], required=True)
    p.add_argument("--exact-weight", type=float, required=True)
    p.add_argument("--expected-model-sha256")
    p.add_argument("--expected-evidencerank-sha256")
    p.add_argument("--expected-scores", type=Path)
    p.add_argument("--tolerance", type=float, default=2e-7)
    p.set_defaults(func=command_stage_a)

    p = sub.add_parser("prepare-test")
    p.add_argument("--candidate-table", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--split-table", required=True, type=Path)
    p.add_argument("--spec-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--declared-split", choices=["test"], required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-peaks", type=int, default=100)
    p.set_defaults(func=command_prepare_test)

    p = sub.add_parser("score-ssr")
    p.add_argument("--candidate-cache", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--expected-checkpoint-sha256", required=True)
    p.add_argument("--out-scores", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=command_score_ssr)

    p = sub.add_parser("fuse")
    p.add_argument("--candidate-cache", required=True, type=Path)
    p.add_argument("--stage-a-scores", required=True, type=Path)
    p.add_argument("--neural-scores", required=True, type=Path)
    p.add_argument("--out-scores", required=True, type=Path)
    p.add_argument("--declared-split", choices=["validation", "test"], required=True)
    p.add_argument("--base-weight", type=float, required=True)
    p.add_argument("--expected-scores", type=Path)
    p.add_argument("--tolerance", type=float, default=2e-7)
    p.set_defaults(func=command_fuse)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
