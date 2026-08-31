#!/usr/bin/env python3
"""Validate and summarize the real-data NPLIB1 ten-spectrum demo bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_smiles(value: str) -> str:
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {value}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def count_positive_peaks(path: Path) -> int:
    count = 0
    in_peaks = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.lower() in {">ms2", ">ms2peaks"}:
            in_peaks = True
            continue
        if line.startswith(">"):
            in_peaks = False
            continue
        if not in_peaks or not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            mz, intensity = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if mz > 0 and intensity > 0:
            count += 1
    return count


def verify_manifest() -> int:
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hash", action="store_true", help="Skip SHA-256 verification")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    checked = 0 if args.skip_hash else verify_manifest()
    with (ROOT / "data" / "labels.tsv").open(newline="", encoding="utf-8") as stream:
        labels = {row["spec"]: row for row in csv.DictReader(stream, delimiter="\t")}
    with (ROOT / "selection.tsv").open(newline="", encoding="utf-8") as stream:
        selection = list(csv.DictReader(stream, delimiter="\t"))
    inference = json.loads(
        (ROOT / "expected" / "demo_inference_results.json").read_text(encoding="utf-8")
    )
    results = {record["spec_id"]: record for record in inference["records"]}
    try:
        zms_cache = torch.load(ROOT / "cache" / "zms_v1.pt", map_location="cpu", weights_only=False)
    except TypeError:
        zms_cache = torch.load(ROOT / "cache" / "zms_v1.pt", map_location="cpu")

    rows = []
    totals = {"1": 0, "5": 0, "10": 0}
    for selected in selection:
        spec_id = selected["spec_id"]
        if spec_id not in labels or spec_id not in results or spec_id not in zms_cache:
            raise RuntimeError(f"Incomplete demo record: {spec_id}")
        spectrum_path = ROOT / "data" / "spec_files" / f"{spec_id}.ms"
        subformula_path = ROOT / "data" / "subformulae" / "default_subformulae" / f"{spec_id}.json"
        peak_count = count_positive_peaks(spectrum_path)
        if peak_count != int(selected["peak_count"]):
            raise AssertionError(f"Peak count mismatch for {spec_id}")
        subformula = json.loads(subformula_path.read_text(encoding="utf-8"))
        if not (subformula.get("output_tbl") or {}).get("formula"):
            raise AssertionError(f"Missing peak formula assignments for {spec_id}")
        embedding = zms_cache[spec_id]
        if tuple(embedding.shape) != (512,) or not torch.isfinite(embedding.float()).all():
            raise AssertionError(f"Invalid Zms embedding for {spec_id}")

        result = results[spec_id]
        truth = canonical_smiles(labels[spec_id]["smiles"])
        candidates = [
            canonical_smiles(item["smiles"]) for item in result["ranked_connected_candidates"]
        ]
        computed_hits = {str(k): truth in candidates[:k] for k in (1, 5, 10)}
        stored_hits = {str(k): bool(result["hits"][str(k)]) for k in (1, 5, 10)}
        if computed_hits != stored_hits:
            raise AssertionError(f"Stored Top-k result mismatch for {spec_id}")
        for key in totals:
            totals[key] += int(computed_hits[key])
        rows.append(
            {
                "order": int(selected["demo_order"]),
                "spec_id": spec_id,
                "formula": labels[spec_id]["formula"],
                "peaks": peak_count,
                "valid_connected": f"{result['n_valid_connected']}/{result['n_generated']}",
                "top1_frequency": result["ranked_connected_candidates"][0]["frequency"],
                "hit_top1": computed_hits["1"],
                "hit_top5": computed_hits["5"],
                "hit_top10": computed_hits["10"],
            }
        )

    print("NPLIB1 real-data demo (curated success cases; not an unbiased benchmark)")
    print("order\tspec_id\tformula\tpeaks\tvalid_conn\ttop1_freq\thit@1/5/10")
    for row in rows:
        flags = "/".join("Y" if row[f"hit_top{k}"] else "N" for k in (1, 5, 10))
        print(
            f"{row['order']}\t{row['spec_id']}\t{row['formula']}\t{row['peaks']}\t"
            f"{row['valid_connected']}\t{row['top1_frequency']}\t{flags}"
        )
    summary = {
        "status": "passed",
        "manifest_files_checked": checked,
        "spectra": len(rows),
        "zms_entries": len(zms_cache),
        "top1": f"{totals['1']}/{len(rows)}",
        "top5": f"{totals['5']}/{len(rows)}",
        "top10": f"{totals['10']}/{len(rows)}",
        "protocol": {
            "n_samples_per_spectrum": inference["protocol"]["n_samples"],
            "n_timesteps": inference["protocol"]["n_timesteps"],
            "condition_source": inference["protocol"]["condition_source"],
            "eval_mode": inference["protocol"]["eval_mode"],
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps({"summary": summary, "records": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
