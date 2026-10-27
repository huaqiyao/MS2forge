#!/usr/bin/env python3
"""Merge fresh DDP candidate caches into a full, BFN-enriched candidate table."""
import argparse
import csv
import glob
import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from utils.compat import SUPPORTED_CANDIDATE_CACHE_SCHEMAS


BFN_COLUMNS = [
    f'{base}_{stat}'
    for base in (
        'mean_top1_prob', 'mean_margin', 'mean_entropy',
        'sum_selected_log_prob', 'mean_selected_log_prob', 'n_edges',
    )
    for stat in ('mean', 'max', 'min')
]


def canonical_noniso(smiles):
    mol = Chem.MolFromSmiles(smiles or '')
    if mol is None:
        return ''
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def structure_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'num_atoms': 0, 'num_bonds': 0, 'ring_count': 0, 'aromatic_bond_count': 0}
    return {
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'aromatic_bond_count': sum(int(bond.GetIsAromatic()) for bond in mol.GetBonds()),
    }


def build_candidates(cache_glob, out_table, out_manifest, summary_json):
    """Expand complete sampling caches without reading or exporting truth."""
    paths = sorted(glob.glob(str(cache_glob)))
    if not paths:
        raise FileNotFoundError(f"No candidate caches match: {cache_glob}")
    rows, manifest, seen = [], [], set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                kind = record.get("type")
                if kind == "batch_done":
                    continue
                if kind not in ("mol_candidates", "input_failure"):
                    raise ValueError(f"Unknown record type at {path}:{number}: {kind}")
                sid = str(record.get("spec_id", "")).strip()
                if not sid or sid in seen:
                    raise ValueError(f"Missing/duplicate spectrum ID at {path}:{number}: {sid}")
                seen.add(sid)
                if kind == "input_failure":
                    manifest.append(dict(spec_id=sid, status=kind, n_candidates=0,
                                         reason=str(record.get("reason", ""))))
                    continue
                if record.get("schema") not in SUPPORTED_CANDIDATE_CACHE_SCHEMAS:
                    raise ValueError(f"Unsupported cache schema for {sid}")
                pred = record["pred"]
                counts = pred["connected_smiles_nonisomeric_counts"]
                n_samples = int(pred["n_generated"])
                if n_samples <= 0 or sum(int(f) for _, f in counts) > n_samples:
                    raise ValueError(f"Invalid draw counts for {sid}")
                stats_map = {x["candidate_canonical_smiles_nonisomeric"]: x
                             for x in pred.get("bfn_stats_nonisomeric", [])}
                first_map = dict(pred.get("candidate_first_seen_nonisomeric", []))
                source_counts = {}
                for isomeric, _ in pred.get("connected_smiles_counts", []):
                    key = canonical_noniso(isomeric)
                    source_counts[key] = source_counts.get(key, 0) + 1
                unique = set()
                for rank, (smiles, freq) in enumerate(counts, 1):
                    canonical = canonical_noniso(smiles)
                    if not canonical or canonical != smiles or "." in smiles or smiles in unique:
                        raise ValueError(f"Invalid/duplicate non-isomeric candidate for {sid}: {smiles}")
                    unique.add(smiles)
                    stat = stats_map.get(smiles, {})
                    missing = [name for name in BFN_COLUMNS if name not in stat]
                    if missing:
                        raise ValueError(f"{sid}: missing BFN posterior statistics; rerun sampling with "
                                         f"the current sampler. Missing: {missing}")
                    values = {name: float(stat[name]) for name in BFN_COLUMNS}
                    if not all(math.isfinite(v) for v in values.values()):
                        raise ValueError(f"Non-finite BFN statistics for {sid}")
                    first = first_map.get(smiles, stat.get("candidate_first_seen"))
                    if first is None or not 1 <= int(first) <= n_samples or int(freq) <= 0:
                        raise ValueError(f"Missing/invalid draw information for {sid}")
                    if "n_draws" in stat and int(stat["n_draws"]) != int(freq):
                        raise ValueError(f"BFN draw count disagrees with candidate frequency for {sid}")
                    rows.append({
                        "spec_id": sid,
                        "candidate_hash": "ms2forge_" + hashlib.sha256(smiles.encode()).hexdigest()[:24],
                        "candidate_rank_freq": rank, "candidate_freq": int(freq),
                        "candidate_freq_fraction": int(freq) / n_samples,
                        "candidate_first_seen": int(first), "n_samples": n_samples,
                        "n_unique": len(counts), "candidate_canonical_smiles": smiles,
                        "candidate_canonical_smiles_nostereo": smiles,
                        "rdkit_valid": 1, "included_reason": "top_rank", "has_bfn_stats_num": 1,
                        "uep_source_edge_candidates": max(1, source_counts.get(smiles, 1)),
                        **structure_features(smiles), **values,
                    })
                manifest.append(dict(spec_id=sid, status="generated" if counts else "no_valid_candidate",
                                     n_candidates=len(counts), reason=""))
    if not manifest:
        raise ValueError("Candidate caches contain no spectrum records")
    fields = ["spec_id", "candidate_hash", "candidate_rank_freq", "candidate_freq",
              "candidate_freq_fraction", "candidate_first_seen", "n_samples", "n_unique",
              "candidate_canonical_smiles", "candidate_canonical_smiles_nostereo", "rdkit_valid",
              "included_reason", "has_bfn_stats_num", "uep_source_edge_candidates", "num_atoms",
              "num_bonds", "ring_count", "aromatic_bond_count", *BFN_COLUMNS]
    rows.sort(key=lambda row: (row["spec_id"], row["candidate_rank_freq"]))
    for path, records, names in [(out_table, rows, fields),
                                 (out_manifest, sorted(manifest, key=lambda x: x["spec_id"]),
                                  ["spec_id", "status", "n_candidates", "reason"])]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=names, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
    summary = {"schema": "ms2forge.full_candidate_table.v2", "truth_columns_read": [],
               "n_spectra": len(manifest), "n_candidates": len(rows),
               "n_without_candidates": sum(x["n_candidates"] == 0 for x in manifest),
               "source_files": paths}
    Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-glob", required=True)
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    print(json.dumps(build_candidates(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
