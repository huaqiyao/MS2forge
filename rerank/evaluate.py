#!/usr/bin/env python3
"""Evaluate reranked candidates with the MS2Forge-UEP-v1.0 metrics.

Uses non-isomeric RDKit canonicalization, Morgan r=2/2048 Tanimoto, and
MassSpecGym MyopicMCES with threshold 15. Samples without a candidate receive
Tanimoto 0 and MCES 100.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


MCES_THRESHOLD = 15
MISSING_MCES = 100.0
TOPK = (1, 10)
_MCES = None


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(true_smiles: str, pred_smiles: str) -> str:
    return hashlib.sha256((true_smiles + "\0" + pred_smiles).encode()).hexdigest()


def canonicalize(smiles: str) -> tuple[str, Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"RDKit parsing failed: {smiles}")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) != 1:
        raise ValueError(f"expected one fragment, found {len(fragments)}")
    key = Chem.MolToSmiles(fragments[0], canonical=True, isomericSmiles=False)
    canonical_mol = Chem.MolFromSmiles(key)
    if canonical_mol is None:
        raise ValueError(f"canonical SMILES failed: {key}")
    return key, canonical_mol


def fingerprint(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=False
    )


def load_truth(split_path: Path, labels_path: Path) -> tuple[list[str], dict[str, str]]:
    with split_path.open(newline="", encoding="utf-8") as handle:
        test_ids = [row["name"] for row in csv.DictReader(handle, delimiter="\t") if row["split"] == "test"]
    wanted = set(test_ids)
    labels: dict[str, str] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        id_col = "identifier" if "identifier" in (rows.fieldnames or []) else "spec"
        for row in rows:
            if row[id_col] in wanted:
                labels[row[id_col]] = canonicalize(row["smiles"])[0]
    missing = wanted.difference(labels)
    if missing:
        raise RuntimeError(f"missing {len(missing)} labels")
    return test_ids, labels


def init_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=120000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS pair_scores (
          pair_key TEXT PRIMARY KEY,
          true_smiles TEXT NOT NULL,
          pred_smiles TEXT NOT NULL,
          tanimoto REAL NOT NULL,
          mces REAL,
          status TEXT NOT NULL,
          error TEXT,
          elapsed_sec REAL,
          updated_at TEXT NOT NULL
        );
        """
    )
    return connection


def import_old_cache(connection: sqlite3.Connection, old_path: Path) -> int:
    if not old_path.exists():
        return 0
    old = sqlite3.connect(old_path)
    cached = {
        row[0]: row[1:]
        for row in old.execute(
            "SELECT pair_key,mces,status,error,elapsed_sec,updated_at FROM pair_scores "
            "WHERE mces IS NOT NULL AND status IN ('ok','exact_match')"
        )
    }
    old.close()
    updates = []
    for (key,) in connection.execute("SELECT pair_key FROM pair_scores WHERE mces IS NULL"):
        value = cached.get(key)
        if value is not None:
            updates.append((*value, key))
    connection.executemany(
        "UPDATE pair_scores SET mces=?,status=?,error=?,elapsed_sec=?,updated_at=? WHERE pair_key=?",
        updates,
    )
    connection.commit()
    return len(updates)


def init_worker() -> None:
    global _MCES
    from functools import partial
    from importlib.metadata import version
    import pulp
    from myopic_mces.myopic_mces import MCES

    if version("myopic-mces") != "1.0.0":
        raise RuntimeError("The frozen evaluation requires myopic-mces==1.0.0; use environment.yml.")
    if not pulp.PULP_CBC_CMD(msg=False).available():
        raise RuntimeError("CBC solver is unavailable; install the environment.yml evaluation dependencies.")
    # Same call and distance field as MassSpecGym 1.3.1's MyopicMCES wrapper.
    # Calling its underlying library avoids changing the training Torch version.
    _MCES = partial(
        MCES,
        ind=0,
        solver="PULP_CBC_CMD",
        threshold=MCES_THRESHOLD,
        always_stronger_bound=True,
        solver_options={"msg": 0, "threads": 1},
    )


def mces_task(task: tuple[str, str, str]):
    key, true_smiles, pred_smiles = task
    started = time.monotonic()
    try:
        distance = float(_MCES(s1=true_smiles, s2=pred_smiles)[1])
        if not np.isfinite(distance) or distance < 0:
            raise ValueError(f"negative MCES distance: {distance}")
        return key, distance, "ok", None, time.monotonic() - started
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return key, None, "error", error, time.monotonic() - started


def compute_pending(connection: sqlite3.Connection, workers: int, progress_path: Path) -> dict:
    tasks = connection.execute(
        "SELECT pair_key,true_smiles,pred_smiles FROM pair_scores WHERE mces IS NULL"
    ).fetchall()
    print(json.dumps({"pending_mces_pairs": len(tasks), "workers": workers}), flush=True)
    started = time.monotonic()
    if not tasks:
        return {"computed": 0, "exceptions": 0, "elapsed_sec": 0.0}
    init_worker()  # Fail once in the parent on version/solver errors, before spawning workers.
    completed = errors = 0
    writes = []
    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=init_worker, maxtasksperchild=500) as pool:
        for key, distance, status, error, elapsed in pool.imap_unordered(mces_task, tasks, chunksize=1):
            completed += 1
            errors += int(status != "ok")
            writes.append((distance, status, error, elapsed, now(), key))
            if len(writes) >= 100:
                connection.executemany(
                    "UPDATE pair_scores SET mces=?,status=?,error=?,elapsed_sec=?,updated_at=? WHERE pair_key=?",
                    writes,
                )
                connection.commit()
                writes.clear()
            if completed % 100 == 0 or completed == len(tasks):
                elapsed_total = time.monotonic() - started
                rate = completed / elapsed_total if elapsed_total else 0.0
                progress = {
                    "updated_at": now(), "completed": completed, "total": len(tasks),
                    "rate_pairs_per_sec": rate, "exceptions": errors,
                    "eta_sec": (len(tasks) - completed) / rate if rate else None,
                }
                progress_path.write_text(json.dumps(progress, indent=2) + "\n")
                if completed % 500 == 0 or completed == len(tasks):
                    print(json.dumps(progress), flush=True)
    if writes:
        connection.executemany(
            "UPDATE pair_scores SET mces=?,status=?,error=?,elapsed_sec=?,updated_at=? WHERE pair_key=?",
            writes,
        )
        connection.commit()
    return {"computed": completed, "exceptions": errors, "elapsed_sec": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-cache", default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--dataset", choices=["msg", "nplib1"], default="nplib1")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids, truth = load_truth(Path(args.split), Path(args.labels))
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise ValueError("Test split must contain unique, nonempty spectrum IDs")

    candidate_columns = [
        "spec_id", "candidate_hash", "candidate_rank_freq",
        "candidate_canonical_smiles_nostereo",
    ]
    candidates = pd.read_csv(
        args.candidate_table, sep="\t", usecols=candidate_columns,
        dtype={"spec_id": str, "candidate_hash": str}, keep_default_na=False,
    )
    scores = pd.read_csv(
        args.scores, sep="\t", dtype={"spec_id": str, "candidate_hash": str},
        keep_default_na=False,
    ).rename(columns={"score": "final_score"})
    frame = candidates.merge(scores, on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    if len(frame) != len(scores) or frame["final_score"].isna().any():
        raise RuntimeError("candidate/score join is incomplete")
    if not np.isfinite(frame["final_score"].astype(float)).all():
        raise RuntimeError("non-finite final score")
    frame = frame.sort_values(
        ["spec_id", "final_score", "candidate_rank_freq"],
        ascending=[True, False, True], kind="mergesort",
    )
    top_rows = {sid: group.head(10).copy() for sid, group in frame.groupby("spec_id", sort=False)}

    connection = init_db(output_dir / "mces_cache.sqlite")
    connection.execute("UPDATE pair_scores SET mces=NULL,status='pending' "
                       "WHERE status NOT IN ('ok','exact_match')")
    samples = []
    for sample_id in test_ids:
        true_smiles = truth[sample_id]
        true_mol = canonicalize(true_smiles)[1]
        true_fp = fingerprint(true_mol)
        group = top_rows.get(sample_id)
        ranked = []
        if group is not None:
            for item in group.itertuples(index=False):
                pred, pred_mol = canonicalize(item.candidate_canonical_smiles_nostereo)
                tanimoto = float(DataStructs.TanimotoSimilarity(true_fp, fingerprint(pred_mol)))
                key = pair_key(true_smiles, pred)
                exact = pred == true_smiles
                connection.execute(
                    "INSERT OR IGNORE INTO pair_scores VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, true_smiles, pred, tanimoto, 0.0 if exact else None,
                     "exact_match" if exact else "pending", None, 0.0 if exact else None, now()),
                )
                ranked.append({
                    "candidate_hash": item.candidate_hash,
                    "smiles": pred,
                    "score": float(item.final_score),
                    "frequency_rank": int(item.candidate_rank_freq),
                    "tanimoto": tanimoto,
                    "pair_key": key,
                })
        samples.append({"sample_id": sample_id, "true_smiles": true_smiles, "ranked": ranked})
    connection.commit()
    reused = import_old_cache(connection, Path(args.old_cache)) if args.old_cache else 0
    print(json.dumps({"old_mces_pairs_reused": reused, "samples": len(samples), "candidate_samples": len(top_rows)}), flush=True)
    run_stats = compute_pending(connection, args.workers, output_dir / "progress.json")
    failures = connection.execute(
        "SELECT pair_key,error FROM pair_scores WHERE mces IS NULL OR status NOT IN ('ok','exact_match')"
    ).fetchall()
    if failures:
        (output_dir / "mces_errors.json").write_text(
            json.dumps([{"pair_key": key, "error": error} for key, error in failures], indent=2),
            encoding="utf-8")
        connection.close()
        raise RuntimeError(f"{len(failures)} MCES pairs failed; no aggregate metrics emitted. See mces_errors.json.")

    pair_mces = dict(connection.execute("SELECT pair_key,mces FROM pair_scores"))
    exact_counts = {k: 0 for k in TOPK}
    tanimoto_sums = {k: 0.0 for k in TOPK}
    mces_sums = {k: 0.0 for k in TOPK}
    per_sample = []
    for sample in samples:
        ranked = sample["ranked"]
        row = {
            "sample_id": sample["sample_id"],
            "true_smiles": sample["true_smiles"],
            "has_valid_candidate": int(bool(ranked)),
            "ranked_candidates_top10": json.dumps([item["smiles"] for item in ranked], separators=(",", ":")),
            "final_scores_top10": json.dumps([item["score"] for item in ranked], separators=(",", ":")),
        }
        for k in TOPK:
            chosen = ranked[:k]
            exact = int(any(item["smiles"] == sample["true_smiles"] for item in chosen))
            tanimoto = max((item["tanimoto"] for item in chosen), default=0.0)
            mces = min((pair_mces[item["pair_key"]] for item in chosen), default=MISSING_MCES)
            exact_counts[k] += exact
            tanimoto_sums[k] += tanimoto
            mces_sums[k] += mces
            row[f"accuracy_top{k}"] = exact
            row[f"tanimoto_top{k}"] = tanimoto
            row[f"mces_top{k}"] = mces
        per_sample.append(row)

    per_sample_path = output_dir / "per_sample_metrics.tsv"
    pd.DataFrame(per_sample).to_csv(per_sample_path, sep="\t", index=False, float_format="%.12g")
    n = len(test_ids)
    summary = {
        "protocol": "MS2Forge-UEP-v1.0-external-reranked",
        "created_at": now(),
        "dataset": args.dataset,
        "N": n,
        "topk": list(TOPK),
        "accuracy": {str(k): {"correct": exact_counts[k], "fraction": exact_counts[k] / n, "percent": 100 * exact_counts[k] / n} for k in TOPK},
        "tanimoto": {str(k): tanimoto_sums[k] / n for k in TOPK},
        "mces": {str(k): mces_sums[k] / n for k in TOPK},
        "coverage": {"candidate_n": len(top_rows), "candidate_fraction": len(top_rows) / n, "no_candidate_n": n - len(top_rows)},
        "mces_solver": {
            "implementation": "myopic_mces.MCES (MassSpecGym-compatible arguments)", "threshold": MCES_THRESHOLD,
            "always_stronger_bound": True, "solver": "PULP_CBC_CMD",
            "workers": args.workers, "old_pairs_reused": reused, **run_stats,
        },
        "input_sha256": {
            "candidate_table": sha256_file(args.candidate_table), "scores": sha256_file(args.scores),
            "split": sha256_file(args.split), "labels": sha256_file(args.labels),
        },
        "output_sha256": {"per_sample_metrics": sha256_file(per_sample_path)},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (output_dir / "mces_exceptions.jsonl").open("w", encoding="utf-8") as handle:
        for values in connection.execute(
            "SELECT pair_key,true_smiles,pred_smiles,mces,error,elapsed_sec FROM pair_scores WHERE status='fallback_exception' ORDER BY pair_key"
        ):
            handle.write(json.dumps(dict(zip(
                ("pair_key", "true_smiles", "pred_smiles", "mces", "error", "elapsed_sec"), values
            )), ensure_ascii=False) + "\n")
    connection.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
