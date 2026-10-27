#!/usr/bin/env python3
"""Build row-aligned UEP Morgan-Tanimoto targets for reranker training.

The output is deliberately a compact sidecar keyed by (spec_id, candidate_hash)
and preserves the exact input row order. Fingerprint settings and molecule
sanitization match MS2Forge unified evaluation protocol v1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from rerank import training_common as tc
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True, type=Path)
    parser.add_argument("--output-targets", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--true-column", default="true_canonical_smiles")
    parser.add_argument("--candidate-column", default="candidate_canonical_smiles")
    parser.add_argument("--truth-column", default="candidate_is_true")
    parser.add_argument("--cache-size", type=int, default=500_000)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-table", required=True)
    parser.add_argument("--declared-split", required=True, choices=["train", "val", "validation"])
    return parser.parse_args()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def make_normalizer(cache_size: int):
    @lru_cache(maxsize=cache_size)
    def normalize(smiles: str) -> tuple[str, Any] | None:
        raw = str(smiles or "").strip()
        if not raw:
            return None
        try:
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(fragments) != 1:
                return None
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            nonstereo = Chem.MolFromSmiles(canonical)
            if nonstereo is None:
                return None
            Chem.SanitizeMol(nonstereo)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                nonstereo, radius=2, nBits=2048, useChirality=False
            )
            return canonical, fingerprint
        except Exception:
            return None

    return normalize


def main() -> None:
    args = parse_args()
    role = 'val' if args.declared_split == 'validation' else args.declared_split
    tc.validate_table_role(args.input_table, tc.split_roles(args.split_table), role)
    for path in (args.output_targets, args.summary_json):
        if path.exists():
            raise ValueError(f'Output already exists: {path}')
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    args.output_targets.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_targets.with_suffix(args.output_targets.suffix + ".tmp")
    normalize = make_normalizer(args.cache_size)

    counts = {
        "rows": 0,
        "exact_rows": 0,
        "invalid_true": 0,
        "invalid_candidate": 0,
        "zero_similarity": 0,
    }
    similarity_sum = 0.0
    started = time.time()
    fields = [
        "spec_id",
        "candidate_hash",
        "candidate_tanimoto_to_truth",
        "candidate_similarity_grade15",
        "candidate_is_true",
    ]

    with args.input_table.open(newline="", encoding="utf-8") as source, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source, delimiter="\t")
        required = {
            "spec_id",
            "candidate_hash",
            args.true_column,
            args.candidate_column,
            args.truth_column,
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Missing required columns in {args.input_table}: {missing}")
        writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        for row in reader:
            if args.limit is not None and counts["rows"] >= args.limit:
                break
            true_record = normalize(str(row.get(args.true_column, "")))
            candidate_record = normalize(str(row.get(args.candidate_column, "")))
            exact = is_true(row.get(args.truth_column))
            if exact:
                counts["exact_rows"] += 1
            if true_record is None:
                counts["invalid_true"] += 1
            if candidate_record is None:
                counts["invalid_candidate"] += 1

            if true_record is None or candidate_record is None:
                raise ValueError(f'Invalid reference/candidate at row {counts["rows"] + 1}')
            if exact != (true_record[0] == candidate_record[0]):
                raise ValueError(f'Exact label disagrees with canonical structures at row {counts["rows"] + 1}')
            if true_record is None or candidate_record is None:
                similarity = 0.0
            elif true_record[0] == candidate_record[0]:
                similarity = 1.0
            else:
                similarity = float(
                    DataStructs.TanimotoSimilarity(true_record[1], candidate_record[1])
                )
            if not math.isfinite(similarity):
                similarity = 0.0
            similarity = min(1.0, max(0.0, similarity))
            if similarity == 0.0:
                counts["zero_similarity"] += 1
            similarity_sum += similarity
            grade = int(round(15.0 * similarity))
            writer.writerow(
                {
                    "spec_id": row["spec_id"],
                    "candidate_hash": row["candidate_hash"],
                    "candidate_tanimoto_to_truth": f"{similarity:.10g}",
                    "candidate_similarity_grade15": grade,
                    "candidate_is_true": int(exact),
                }
            )
            counts["rows"] += 1
            if args.progress_every and counts["rows"] % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"[targets] rows={counts['rows']:,} rate={counts['rows'] / elapsed:,.1f}/s "
                    f"mean_tanimoto={similarity_sum / counts['rows']:.4f}",
                    flush=True,
                )

    os.replace(tmp_path, args.output_targets)
    elapsed = time.time() - started
    cache_info = normalize.cache_info()
    summary = {
        "schema": "ms2forge.similarity_targets.v1",
        "status": "experimental_not_frozen",
        "input_table": str(args.input_table),
        "output_targets": str(args.output_targets),
        "target_definition": {
            "molecule": "sanitized single-fragment non-stereochemical canonical SMILES",
            "fingerprint": "RDKit Morgan bit vector",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": False,
            "similarity": "TanimotoSimilarity",
            "grade": "round(15 * similarity)",
        },
        "columns": {
            "true": args.true_column,
            "candidate": args.candidate_column,
            "exact": args.truth_column,
        },
        "counts": counts,
        "mean_candidate_tanimoto": similarity_sum / counts["rows"] if counts["rows"] else 0.0,
        "elapsed_seconds": elapsed,
        "rows_per_second": counts["rows"] / elapsed if elapsed > 0 else None,
        "fingerprint_cache": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        },
        "row_alignment": "same order as input; keys retained and checked by trainer",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
