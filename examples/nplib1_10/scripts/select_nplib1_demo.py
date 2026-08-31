#!/usr/bin/env python3
"""Select ten high-quality, structurally diverse NPLIB1 test success cases."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


SUPPORTED_ELEMENTS = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def read_spectrum(path: Path) -> dict:
    peaks: list[tuple[float, float]] = []
    metadata: dict[str, str] = {}
    in_peaks = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() in {">ms2", ">ms2peaks"}:
            in_peaks = True
            continue
        if line.startswith(">"):
            in_peaks = False
            parts = line[1:].split(maxsplit=1)
            if len(parts) == 2:
                metadata[parts[0].lower()] = parts[1]
            continue
        if line.startswith("#"):
            parts = line[1:].split(maxsplit=1)
            if len(parts) == 2:
                metadata[parts[0].lower()] = parts[1]
            continue
        if not in_peaks:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            mz, intensity = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if mz > 0 and intensity > 0 and math.isfinite(mz) and math.isfinite(intensity):
            peaks.append((mz, intensity))
    unique_mz = len({round(mz, 6) for mz, _ in peaks})
    return {
        "peak_count": len(peaks),
        "unique_mz_count": unique_mz,
        "base_peak_intensity": max((intensity for _, intensity in peaks), default=0.0),
        "library_quality": metadata.get("libraryqualitystring", ""),
        "compound_name": metadata.get("compound", ""),
    }


def formula_supported(formula: str) -> bool:
    matches = FORMULA_PATTERN.findall(str(formula))
    return bool(matches) and all(element in SUPPORTED_ELEMENTS or element == "H" for element, _ in matches)


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    data_root = workspace / "data" / "nplib1"
    metrics = pd.read_csv(workspace / "evaluation" / "uep" / "per_sample_metrics.tsv", sep="\t")
    labels = pd.read_csv(data_root / "labels.tsv", sep="\t", dtype=str, keep_default_na=False)
    splits = pd.read_csv(data_root / "split.tsv", sep="\t", dtype=str)
    labels = labels.merge(splits.rename(columns={"name": "spec"}), on="spec", how="inner")
    labels = labels[labels["split"].eq("test")].drop_duplicates("spec")
    frame = labels.merge(metrics, left_on="spec", right_on="sample_id", how="inner")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        if int(record["accuracy_top1"]) != 1:
            continue
        if int(record["has_inference"]) != 1 or int(record["has_valid_candidate"]) != 1:
            continue
        if not formula_supported(record["formula"]):
            continue
        if "H" not in str(record["ionization"]).replace(" ", ""):
            continue
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            continue
        spectrum_path = data_root / "spec_files" / f"{record['spec']}.ms"
        if not spectrum_path.is_file():
            continue
        spectrum = read_spectrum(spectrum_path)
        if spectrum["peak_count"] < 40 or spectrum["unique_mz_count"] < 35:
            continue
        connected_ratio = float(record["n_valid_connected"]) / max(int(record["n_generated"]), 1)
        if connected_ratio < 0.80:
            continue
        candidates = json.loads(record["ranked_candidates_top10"])
        frequencies = json.loads(record["frequency_counts_top10"])
        top1_frequency = int(frequencies[0]) if frequencies else 0
        top1_fraction = top1_frequency / max(int(record["n_generated"]), 1)
        known_instrument = not any(
            token in str(record["instrument"]).lower() for token in ("unknown", "none", "n/a")
        )
        gold = spectrum["library_quality"].lower() == "gold"
        quality = (
            0.40 * connected_ratio
            + 0.32 * min(top1_fraction / 0.50, 1.0)
            + 0.13 * min(math.log1p(spectrum["unique_mz_count"]) / math.log1p(300), 1.0)
            + 0.10 * float(known_instrument)
            + 0.05 * float(gold)
        )
        rows.append(
            {
                "spec_id": record["spec"],
                "compound_name": spectrum["compound_name"] or record["name"],
                "formula": record["formula"],
                "ionization": record["ionization"],
                "instrument": record["instrument"],
                "smiles": record["smiles"],
                "inchikey": record["inchikey"],
                "heavy_atom_count": heavy_atom_count(mol),
                "peak_count": spectrum["peak_count"],
                "unique_mz_count": spectrum["unique_mz_count"],
                "library_quality": spectrum["library_quality"],
                "n_generated": int(record["n_generated"]),
                "n_valid_connected": int(record["n_valid_connected"]),
                "valid_connected_fraction": connected_ratio,
                "top1_frequency": top1_frequency,
                "top1_frequency_fraction": top1_fraction,
                "predicted_top1_smiles": candidates[0],
                "predicted_top10_smiles": candidates,
                "quality_score": quality,
                "fingerprint": generator.GetFingerprint(mol),
            }
        )

    # Keep one spectrum per structure, choosing the best spectrum/most stable prediction.
    best_by_structure: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (-item["quality_score"], item["spec_id"])):
        best_by_structure.setdefault(row["inchikey"], row)
    pool = list(best_by_structure.values())
    if len(pool) < args.count:
        raise RuntimeError(f"Only {len(pool)} eligible unique structures for requested {args.count}")

    # Quality-diversity greedy selection. The first item is the strongest quality case;
    # later picks balance quality with distance from structures already selected.
    qualities = np.asarray([row["quality_score"] for row in pool], dtype=float)
    q_min, q_max = float(qualities.min()), float(qualities.max())
    for row in pool:
        row["quality_normalized"] = (row["quality_score"] - q_min) / max(q_max - q_min, 1e-12)
    pool.sort(key=lambda item: (-item["quality_score"], item["spec_id"]))
    selected = [pool.pop(0)]
    while len(selected) < args.count:
        best_index = None
        best_value = -1.0
        for index, row in enumerate(pool):
            max_similarity = max(
                DataStructs.TanimotoSimilarity(row["fingerprint"], chosen["fingerprint"])
                for chosen in selected
            )
            diversity = 1.0 - max_similarity
            size_bonus = 0.05 if not any(
                abs(row["heavy_atom_count"] - chosen["heavy_atom_count"]) <= 2 for chosen in selected
            ) else 0.0
            value = 0.68 * row["quality_normalized"] + 0.32 * diversity + size_bonus
            if value > best_value:
                best_index, best_value = index, value
        chosen = pool.pop(int(best_index))
        chosen["selection_value"] = best_value
        selected.append(chosen)

    selected[0]["selection_value"] = selected[0]["quality_normalized"]
    for index, row in enumerate(selected, start=1):
        row["demo_order"] = index
        row["max_tanimoto_to_earlier"] = (
            0.0
            if index == 1
            else max(
                DataStructs.TanimotoSimilarity(row["fingerprint"], earlier["fingerprint"])
                for earlier in selected[: index - 1]
            )
        )
    for row in selected:
        row.pop("fingerprint")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "demo_order", "spec_id", "compound_name", "formula", "ionization", "instrument",
        "inchikey", "smiles", "predicted_top1_smiles", "heavy_atom_count", "peak_count",
        "unique_mz_count", "library_quality", "n_generated", "n_valid_connected",
        "valid_connected_fraction", "top1_frequency", "top1_frequency_fraction",
        "quality_score", "max_tanimoto_to_earlier", "selection_value",
    ]
    pd.DataFrame(selected)[columns].to_csv(output, sep="\t", index=False, float_format="%.6g")
    payload = {
        "schema": "ms2forge.nplib1_curated_demo.v1",
        "selection_scope": "official NPLIB1 canopus_hplus_100_0 test split",
        "selection_type": "curated success cases; not an unbiased evaluation subset",
        "eligibility": {
            "top1_exact_match": True,
            "unique_structure_by_inchikey": True,
            "supported_formula_elements": sorted(SUPPORTED_ELEMENTS),
            "minimum_positive_peaks": 40,
            "minimum_unique_mz": 35,
            "minimum_connected_candidate_fraction": 0.80,
        },
        "selected": [
            {
                key: value
                for key, value in row.items()
                if key not in {"predicted_top10_smiles"}
            }
            | {"predicted_top10_smiles": row["predicted_top10_smiles"]}
            for row in selected
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(selected)[columns].to_string(index=False))
    print(f"\nWrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
