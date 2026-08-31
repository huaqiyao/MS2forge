#!/usr/bin/env python3
"""Convert MS2Forge candidate-cache JSONL into a label-free rerank TSV.

``scripts/sample.py`` writes one ``flash_candidate_cache.v1`` molecule record
per spectrum plus ``batch_done`` markers.  EvidenceRank-lite and the SSR cache
preparation commands consume a row-wise TSV instead.  This bridge expands the
connected, canonical candidate counts without exporting truth labels or hit
indicators.

The resulting table is directly compatible with:

* ``ms2forge_evidencerank_lite_features.py --candidate-structures``;
* ``ms2forge_spectrum_structure_ranker.py prepare-candidates``; and
* ``ms2forge_apply_frozen_test.py prepare-test``.

It is deliberately *not* advertised as a complete frozen Stage-A feature
table.  The compact sampling cache does not contain every historical BFN,
mass-fragment and group-relative feature used by ``stage_a_model.pkl``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA = "flash_candidate_cache.v1"
OUTPUT_SCHEMA = "ms2forge.rerank_candidate_table.v1"

FIELDNAMES = [
    "spec_id",
    "candidate_hash",
    "candidate_rank_freq",
    "candidate_freq",
    "candidate_freq_fraction",
    "candidate_first_seen",
    "n_samples",
    "n_unique",
    "candidate_canonical_smiles",
    "candidate_canonical_smiles_nostereo",
    "candidate_uep_identity",
    "rdkit_valid",
    "included_reason",
    "n_valid",
    "n_valid_connected",
    "n_invalid",
    "n_disconnected",
    "source_rank",
    "source_world_size",
    "source_batch_idx",
    "source_jsonl",
    "source_line",
]


class CandidateBridgeError(RuntimeError):
    """Raised when an input cache violates the bridge contract."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_hash(smiles: str, prefix: str = "ms2forge") -> str:
    """Return a stable structure-level key for cross-table joins."""
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _positive_int(value: object, label: str, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateBridgeError(f"{source}: invalid {label}={value!r}") from exc
    if parsed < 0:
        raise CandidateBridgeError(f"{source}: {label} must be non-negative, got {parsed}")
    return parsed


def _candidate_counts(pred: dict, source: str) -> List[Tuple[str, int]]:
    raw = pred.get("connected_smiles_nonisomeric_counts")
    if raw is None:
        raw = pred.get("connected_smiles_counts")
    if raw is None:
        raise CandidateBridgeError(
            f"{source}: pred lacks connected_smiles_nonisomeric_counts"
        )
    if not isinstance(raw, list):
        raise CandidateBridgeError(f"{source}: candidate counts must be a list")

    merged: "OrderedDict[str, int]" = OrderedDict()
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise CandidateBridgeError(
                f"{source}: candidate count #{index} must be [smiles, count]"
            )
        smiles = str(item[0]).strip()
        count = _positive_int(item[1], "candidate count", source)
        if not smiles or count == 0:
            continue
        merged[smiles] = merged.get(smiles, 0) + count

    # Counter.most_common() already orders the source list.  Re-sorting after
    # merging repeated strings keeps frequency order while retaining the first
    # source occurrence as the deterministic tie-breaker.
    first_position = {smiles: index for index, smiles in enumerate(merged)}
    return sorted(
        merged.items(),
        key=lambda item: (-item[1], first_position[item[0]]),
    )


def _iter_records(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CandidateBridgeError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise CandidateBridgeError(
                    f"{path}:{line_number}: JSON value must be an object"
                )
            yield line_number, record


def export_candidates(
    input_paths: Sequence[Path],
    output_path: Path,
    summary_path: Path,
    hash_prefix: str = "ms2forge",
    overwrite: bool = False,
) -> dict:
    if not input_paths:
        raise CandidateBridgeError("at least one --input JSONL is required")
    if not hash_prefix or any(char.isspace() for char in hash_prefix):
        raise CandidateBridgeError("--hash-prefix must be non-empty and contain no whitespace")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"candidate cache not found: {path}")
    for path in (output_path, summary_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    output_temp = output_path.with_name(output_path.name + ".tmp")
    summary_temp = summary_path.with_name(summary_path.name + ".tmp")
    for temp in (output_temp, summary_temp):
        if temp.exists():
            temp.unlink()

    seen_specs: Dict[str, str] = {}
    hash_registry: Dict[str, str] = {}
    n_source_records = 0
    n_batch_markers = 0
    n_candidates = 0
    n_empty_spectra = 0

    try:
        with output_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
            writer.writeheader()

            for input_path in input_paths:
                for line_number, record in _iter_records(input_path):
                    source = f"{input_path}:{line_number}"
                    if record.get("schema") != SCHEMA:
                        raise CandidateBridgeError(
                            f"{source}: expected schema={SCHEMA!r}, got {record.get('schema')!r}"
                        )
                    record_type = record.get("type")
                    if record_type == "batch_done":
                        n_batch_markers += 1
                        continue
                    if record_type != "mol_candidates":
                        raise CandidateBridgeError(
                            f"{source}: unsupported record type {record_type!r}"
                        )

                    spec_id = str(record.get("spec_id", "")).strip()
                    if not spec_id:
                        raise CandidateBridgeError(f"{source}: missing spec_id")
                    if spec_id in seen_specs:
                        raise CandidateBridgeError(
                            f"{source}: duplicate spec_id={spec_id!r}; first seen at {seen_specs[spec_id]}"
                        )
                    seen_specs[spec_id] = source
                    n_source_records += 1

                    pred = record.get("pred")
                    if not isinstance(pred, dict):
                        raise CandidateBridgeError(f"{source}: pred must be an object")
                    counts = _candidate_counts(pred, source)
                    if not counts:
                        n_empty_spectra += 1

                    n_samples = _positive_int(pred.get("n_generated", 0), "n_generated", source)
                    n_unique = len(counts)
                    diagnostics = {
                        "n_valid": _positive_int(pred.get("n_valid", 0), "n_valid", source),
                        "n_valid_connected": _positive_int(
                            pred.get("n_valid_connected", 0), "n_valid_connected", source
                        ),
                        "n_invalid": _positive_int(pred.get("n_invalid", 0), "n_invalid", source),
                        "n_disconnected": _positive_int(
                            pred.get("n_disconnected", 0), "n_disconnected", source
                        ),
                    }

                    for rank_index, (smiles, frequency) in enumerate(counts, start=1):
                        key = candidate_hash(smiles, hash_prefix)
                        previous = hash_registry.get(key)
                        if previous is not None and previous != smiles:
                            raise CandidateBridgeError(
                                f"{source}: candidate hash collision between {previous!r} and {smiles!r}"
                            )
                        hash_registry[key] = smiles
                        writer.writerow(
                            {
                                "spec_id": spec_id,
                                "candidate_hash": key,
                                "candidate_rank_freq": rank_index,
                                "candidate_freq": frequency,
                                "candidate_freq_fraction": (
                                    f"{frequency / n_samples:.12g}" if n_samples else ""
                                ),
                                # The compact cache stores counts, not the
                                # original draw index.  Do not fabricate it.
                                "candidate_first_seen": "",
                                "n_samples": n_samples,
                                "n_unique": n_unique,
                                "candidate_canonical_smiles": smiles,
                                "candidate_canonical_smiles_nostereo": smiles,
                                "candidate_uep_identity": smiles,
                                "rdkit_valid": "true",
                                "included_reason": "top_rank",
                                **diagnostics,
                                "source_rank": record.get("rank", ""),
                                "source_world_size": record.get("world_size", ""),
                                "source_batch_idx": record.get("batch_idx", ""),
                                "source_jsonl": str(input_path),
                                "source_line": line_number,
                            }
                        )
                        n_candidates += 1

        os.replace(output_temp, output_path)

        summary = {
            "schema": OUTPUT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": "label-free export from generated candidates; truth and hit fields excluded",
            "input_files": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in input_paths
            ],
            "output_tsv": str(output_path),
            "output_sha256": sha256_file(output_path),
            "n_source_records": n_source_records,
            "n_batch_markers": n_batch_markers,
            "n_spectra": len(seen_specs),
            "n_empty_spectra": n_empty_spectra,
            "n_candidates": n_candidates,
            "candidate_identity": "canonical non-isomeric SMILES from connected valid predictions",
            "candidate_hash": f"{hash_prefix}_ + first 24 hex of SHA256(candidate identity)",
            "truth_fields_exported": [],
            "compatible_consumers": [
                "EvidenceRank-lite --candidate-structures",
                "SSR prepare-candidates",
                "SSR prepare-test",
            ],
            "frozen_stage_a_ready": False,
            "stage_a_boundary": (
                "The sampling JSONL does not contain every historical BFN, mass-fragment and "
                "group-relative feature required by stage_a_model.pkl. Merge/compute those "
                "features and EvidenceRank scores before frozen Stage-A inference."
            ),
        }
        summary_temp.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(summary_temp, summary_path)
        return summary
    except Exception:
        for temp in (output_temp, summary_temp):
            if temp.exists():
                temp.unlink()
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more rank-specific flash_candidate_cache.v1 JSONL files.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output candidate TSV.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Audit summary; defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--hash-prefix",
        default="ms2forge",
        help="Prefix for deterministic candidate hashes (default: ms2forge).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_paths = [path.expanduser().resolve() for path in args.input]
    output_path = args.output.expanduser().resolve()
    summary_path = (
        args.summary_json.expanduser().resolve()
        if args.summary_json is not None
        else output_path.with_name(output_path.name + ".summary.json")
    )
    summary = export_candidates(
        input_paths=input_paths,
        output_path=output_path,
        summary_path=summary_path,
        hash_prefix=args.hash_prefix,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
