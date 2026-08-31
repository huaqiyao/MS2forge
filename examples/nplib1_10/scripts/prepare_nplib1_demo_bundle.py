#!/usr/bin/env python3
"""Build a minimal ten-spectrum NPLIB1 demo bundle from a frozen selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--selection-tsv", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    data_root = workspace / "data" / "nplib1"
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")

    selection = pd.read_csv(args.selection_tsv, sep="\t", dtype={"spec_id": str})
    selected_ids = selection["spec_id"].tolist()
    if len(selected_ids) != 10 or len(set(selected_ids)) != 10:
        raise ValueError("Selection must contain exactly ten unique spectrum IDs")

    labels = pd.read_csv(data_root / "labels.tsv", sep="\t", dtype=str, keep_default_na=False)
    labels = labels[labels["spec"].isin(selected_ids)].drop_duplicates("spec").copy()
    labels["_order"] = labels["spec"].map({sid: index for index, sid in enumerate(selected_ids)})
    labels = labels.sort_values("_order").drop(columns="_order")
    if labels["spec"].tolist() != selected_ids:
        raise RuntimeError("Could not recover all selected labels in frozen order")

    spec_dir = output / "data" / "spec_files"
    subformula_dir = output / "data" / "subformulae" / "default_subformulae"
    cache_dir = output / "cache"
    expected_dir = output / "expected"
    for directory in (spec_dir, subformula_dir, cache_dir, expected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels.to_csv(output / "data" / "labels.tsv", sep="\t", index=False)
    pd.DataFrame({"name": selected_ids, "split": ["test"] * len(selected_ids)}).to_csv(
        output / "data" / "split.tsv", sep="\t", index=False
    )

    for spec_id in selected_ids:
        shutil.copy2(data_root / "spec_files" / f"{spec_id}.ms", spec_dir / f"{spec_id}.ms")
        shutil.copy2(
            data_root / "subformulae" / "default_subformulae" / f"{spec_id}.json",
            subformula_dir / f"{spec_id}.json",
        )

    full_cache = torch.load(workspace / "cache" / "nplib1" / "zms_v1.pt", map_location="cpu")
    mini_cache = {spec_id: full_cache[spec_id].clone() for spec_id in selected_ids}
    if set(mini_cache) != set(selected_ids):
        raise RuntimeError("Selected Zms cache is incomplete")
    torch.save(mini_cache, cache_dir / "zms_v1.pt")

    selection.to_csv(output / "selection.tsv", sep="\t", index=False)
    shutil.copy2(args.selection_json, output / "selection.json")

    metrics = pd.read_csv(
        workspace / "evaluation" / "uep" / "per_sample_metrics.tsv",
        sep="\t",
        dtype={"sample_id": str},
    )
    metrics = metrics[metrics["sample_id"].isin(selected_ids)].copy()
    metrics["_order"] = metrics["sample_id"].map(
        {sid: index for index, sid in enumerate(selected_ids)}
    )
    metrics = metrics.sort_values("_order").drop(columns="_order")
    metrics.to_csv(expected_dir / "frozen_per_sample_metrics.tsv", sep="\t", index=False)

    selection_payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    provenance = {
        "schema": "ms2forge.nplib1_10_demo_bundle.v1",
        "source_dataset": "NPLIB1 official canopus_hplus_100_0 test split",
        "source_workspace": str(workspace),
        "source_full_test_metrics": str(workspace / "evaluation" / "uep" / "summary.json"),
        "selection": selection_payload,
        "included": {
            "spectra": len(list(spec_dir.glob("*.ms"))),
            "subformula_files": len(list(subformula_dir.glob("*.json"))),
            "labels": len(labels),
            "zms_cache_entries": len(mini_cache),
        },
        "excluded": [
            "full NPLIB1 dataset",
            "model checkpoints",
            "training data",
            "full embedding caches",
            "full candidate caches",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest_lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        manifest_lines.append(f"{sha256(path)}  {relative}")
    (output / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(json.dumps({"status": "built", "output": str(output), "files": len(manifest_lines) + 1,
                      "bytes": total_bytes}, indent=2))


if __name__ == "__main__":
    main()
