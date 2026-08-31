#!/usr/bin/env python3
"""Train-only spectrum--structure compatibility reranker for MS2Forge.

This branch deliberately does not load the historical ``align.pt`` checkpoint:
its provenance audit names the test split.  Instead, a small spectrum encoder is
trained from scratch against Morgan fingerprints using train-only truth/hard-
negative pairs.  Validation candidate scores can then be fused with an existing
EvidenceRank score under exact-match non-inferiority constraints.

The script never contains a test path and refuses split names other than train
for pair preparation and val/valid/validation for candidate preparation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "B", "Si", "As", "Se")
ION_TYPES = ("[M+H]+", "[M-H]-", "[M+Na]+")
INSTRUMENT_TYPES = ("orbitrap", "qtof", "fticr", "other")
PEAK_DIM = 5
META_DIM = len(ELEMENTS) + 1 + len(ION_TYPES) + len(INSTRUMENT_TYPES) + 1
FP_BITS = 2048
FP_BYTES = FP_BITS // 8
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_instrument(value: str) -> int:
    text = str(value).lower().replace("-", "").replace("_", "")
    if "orbitrap" in text:
        return 0
    if "qtof" in text or "tof" in text:
        return 1
    if "fticr" in text or "fourier" in text:
        return 2
    return 3


def formula_counts(formula: str) -> np.ndarray:
    counts = {element: 0.0 for element in ELEMENTS}
    for element, number in FORMULA_RE.findall(str(formula)):
        if element in counts:
            counts[element] += float(number or 1)
    return np.asarray([math.log1p(counts[e]) / 5.0 for e in ELEMENTS], dtype=np.float32)


def parse_ms_file(path: Path, max_peaks: int) -> tuple[np.ndarray, float]:
    parent_mass = 0.0
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">parentmass"):
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        parent_mass = float(fields[1])
                    except ValueError:
                        parent_mass = 0.0
                continue
            if line.startswith(">ms2peaks"):
                in_peaks = True
                continue
            if not in_peaks or line.startswith((">", "#")):
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

    peaks.sort(key=lambda pair: pair[1], reverse=True)
    peaks = peaks[:max_peaks]
    max_intensity = max((pair[1] for pair in peaks), default=1.0)
    parent_mass = parent_mass if parent_mass > 0 else max((pair[0] for pair in peaks), default=1000.0)
    encoded = np.zeros((max_peaks, PEAK_DIM), dtype=np.float32)
    for rank, (mz, intensity) in enumerate(peaks):
        rel_i = min(max(intensity / max_intensity, 0.0), 1.0)
        encoded[rank] = (
            min(mz / 1000.0, 2.0),
            min(mz / max(parent_mass, 1e-6), 2.0),
            min(max(parent_mass - mz, 0.0) / 1000.0, 2.0),
            math.sqrt(rel_i),
            1.0 / math.sqrt(rank + 1.0),
        )
    return encoded, float(parent_mass)


def build_spectrum_arrays(
    spec_ids: list[str], labels: pd.DataFrame, spec_dir: Path, max_peaks: int, out_dir: Path
) -> dict:
    label_by_spec = labels.set_index("spec", drop=False)
    peaks_mm = np.lib.format.open_memmap(
        out_dir / "spectra_peaks.npy", mode="w+", dtype=np.float16,
        shape=(len(spec_ids), max_peaks, PEAK_DIM),
    )
    meta_mm = np.lib.format.open_memmap(
        out_dir / "spectra_meta.npy", mode="w+", dtype=np.float32,
        shape=(len(spec_ids), META_DIM),
    )
    missing_files = 0
    missing_labels = 0
    for index, spec_id in enumerate(spec_ids):
        if spec_id not in label_by_spec.index:
            missing_labels += 1
            peaks = np.zeros((max_peaks, PEAK_DIM), dtype=np.float32)
            parent_mass = 0.0
            row = None
        else:
            row = label_by_spec.loc[spec_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            path = spec_dir / f"{spec_id}.ms"
            if path.exists():
                peaks, parent_mass = parse_ms_file(path, max_peaks)
            else:
                missing_files += 1
                peaks = np.zeros((max_peaks, PEAK_DIM), dtype=np.float32)
                parent_mass = 0.0
        peaks_mm[index] = peaks.astype(np.float16)
        meta = np.zeros(META_DIM, dtype=np.float32)
        if row is not None:
            meta[: len(ELEMENTS)] = formula_counts(row.get("formula", ""))
            offset = len(ELEMENTS)
            meta[offset] = min(parent_mass / 1000.0, 2.0)
            ion = str(row.get("ionization", ""))
            if ion in ION_TYPES:
                meta[offset + 1 + ION_TYPES.index(ion)] = 1.0
            inst_offset = offset + 1 + len(ION_TYPES)
            meta[inst_offset + normalize_instrument(row.get("instrument", ""))] = 1.0
            meta[-1] = float(np.count_nonzero(peaks[:, 3])) / max_peaks
        meta_mm[index] = meta
    peaks_mm.flush()
    meta_mm.flush()
    return {"missing_spectrum_files": missing_files, "missing_label_rows": missing_labels}


_FP_GENERATOR = None


def fingerprint_packed(smiles: str) -> tuple[np.ndarray, bool]:
    global _FP_GENERATOR
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if _FP_GENERATOR is None:
        _FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FP_BITS, includeChirality=False
        )
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FP_BYTES, dtype=np.uint8), False
    fp = _FP_GENERATOR.GetFingerprint(mol)
    bits = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return np.packbits(bits, bitorder="big"), True


def _fingerprint_pair(pair: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    chosen, rejected = pair
    chosen_fp, chosen_ok = fingerprint_packed(chosen)
    rejected_fp, rejected_ok = fingerprint_packed(rejected)
    return chosen_fp, rejected_fp, chosen_ok, rejected_ok


def parallel_map(function, items: Iterable, workers: int, chunksize: int = 128):
    if workers <= 1:
        yield from map(function, items)
        return
    context = mp.get_context("spawn")
    with context.Pool(workers) as pool:
        yield from pool.imap(function, items, chunksize=chunksize)


def load_labels(path: Path) -> pd.DataFrame:
    usecols = ["spec", "ionization", "formula", "smiles", "instrument"]
    return pd.read_csv(path, sep="\t", usecols=usecols, dtype=str, keep_default_na=False)


def verify_split(spec_ids: Iterable[str], split_path: Path, allowed: set[str]) -> dict:
    split = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False)
    mapping = dict(zip(split["name"], split["split"]))
    counts: dict[str, int] = {}
    missing = 0
    violations: list[tuple[str, str]] = []
    for spec_id in spec_ids:
        value = mapping.get(spec_id)
        if value is None:
            missing += 1
            continue
        counts[value] = counts.get(value, 0) + 1
        if value not in allowed and len(violations) < 20:
            violations.append((spec_id, value))
    if violations or missing:
        raise RuntimeError(
            f"split audit failed: missing={missing}, violations={violations[:5]}, counts={counts}"
        )
    return counts


def command_prepare_train(args: argparse.Namespace) -> None:
    if args.declared_split.lower() != "train":
        raise ValueError("prepare-train requires --declared-split train")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str, keep_default_na=False)
    if args.limit is not None:
        pairs = pairs.head(args.limit).copy()
    required = {
        "spec_id", "chosen_smiles", "chosen_candidate_hash", "rejected_smiles",
        "rejected_candidate_hash", "rejected_tanimoto",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"preference table missing columns: {sorted(missing)}")
    if pairs["spec_id"].duplicated().any():
        # Multiple negatives are valid, but spectra are stored once and indexed.
        pass
    spec_ids = pairs["spec_id"].drop_duplicates().tolist()
    split_counts = verify_split(spec_ids, Path(args.split_table), {"train"})
    labels = load_labels(Path(args.labels))
    spectra_audit = build_spectrum_arrays(
        spec_ids, labels, Path(args.spec_dir), args.max_peaks, out_dir
    )
    spec_to_index = {spec_id: index for index, spec_id in enumerate(spec_ids)}
    np.save(out_dir / "pair_spec_index.npy", pairs["spec_id"].map(spec_to_index).to_numpy(np.int32))
    chosen_group, uniques = pd.factorize(pairs["chosen_candidate_hash"], sort=True)
    np.save(out_dir / "chosen_group.npy", chosen_group.astype(np.int32))
    np.save(out_dir / "rejected_tanimoto.npy", pd.to_numeric(
        pairs["rejected_tanimoto"], errors="coerce"
    ).fillna(0.0).clip(0, 1).to_numpy(np.float32))

    chosen_mm = np.lib.format.open_memmap(
        out_dir / "chosen_fp_packed.npy", mode="w+", dtype=np.uint8,
        shape=(len(pairs), FP_BYTES),
    )
    rejected_mm = np.lib.format.open_memmap(
        out_dir / "rejected_fp_packed.npy", mode="w+", dtype=np.uint8,
        shape=(len(pairs), FP_BYTES),
    )
    invalid_chosen = 0
    invalid_rejected = 0
    iterable = zip(pairs["chosen_smiles"].tolist(), pairs["rejected_smiles"].tolist())
    for index, (chosen_fp, rejected_fp, chosen_ok, rejected_ok) in enumerate(
        parallel_map(_fingerprint_pair, iterable, args.workers)
    ):
        chosen_mm[index] = chosen_fp
        rejected_mm[index] = rejected_fp
        invalid_chosen += int(not chosen_ok)
        invalid_rejected += int(not rejected_ok)
        if index and index % 10000 == 0:
            print(json.dumps({"fingerprinted_pairs": index, "total": len(pairs)}), flush=True)
    chosen_mm.flush()
    rejected_mm.flush()

    pd.DataFrame({"spec_id": spec_ids}).to_csv(out_dir / "spectra.tsv", sep="\t", index=False)
    summary = {
        "schema": "ms2forge.ssr_train_cache.v1",
        "status": "experimental_not_frozen",
        "source_split": "train_only",
        "test_access": "closed",
        "n_pairs": int(len(pairs)),
        "n_spectra": int(len(spec_ids)),
        "n_unique_truth_structures": int(len(uniques)),
        "invalid_chosen_fingerprints": invalid_chosen,
        "invalid_rejected_fingerprints": invalid_rejected,
        "split_counts": split_counts,
        "spectra_audit": spectra_audit,
        "max_peaks": args.max_peaks,
        "peak_dim": PEAK_DIM,
        "meta_dim": META_DIM,
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": FP_BITS, "chirality": False},
        "source_sha256": sha256_file(Path(args.pairs)),
    }
    json_dump(summary, out_dir / "prepare_summary.json")
    (out_dir / "TRAIN_CACHE_READY_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def _fingerprint_single(smiles: str) -> tuple[np.ndarray, bool]:
    return fingerprint_packed(smiles)


def command_prepare_candidates(args: argparse.Namespace) -> None:
    if args.declared_split.lower() not in {"val", "valid", "validation"}:
        raise ValueError("candidate preparation is validation-only; test is sealed")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["spec_id", "candidate_hash", "candidate_canonical_smiles", "candidate_rank_freq"]
    candidates = pd.read_csv(
        args.candidate_table, sep="\t", usecols=usecols, dtype=str, keep_default_na=False
    )
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()
    if candidates.duplicated(["spec_id", "candidate_hash"]).any():
        raise RuntimeError("duplicate (spec_id, candidate_hash) rows in validation candidates")
    spec_ids = candidates["spec_id"].drop_duplicates().tolist()
    split_counts = verify_split(spec_ids, Path(args.split_table), {"val"})
    labels = load_labels(Path(args.labels))
    spectra_audit = build_spectrum_arrays(
        spec_ids, labels, Path(args.spec_dir), args.max_peaks, out_dir
    )
    spec_to_index = {spec_id: index for index, spec_id in enumerate(spec_ids)}
    np.save(out_dir / "candidate_spec_index.npy", candidates["spec_id"].map(spec_to_index).to_numpy(np.int32))
    packed_mm = np.lib.format.open_memmap(
        out_dir / "candidate_fp_packed.npy", mode="w+", dtype=np.uint8,
        shape=(len(candidates), FP_BYTES),
    )
    invalid = 0
    for index, (fingerprint, ok) in enumerate(parallel_map(
        _fingerprint_single, candidates["candidate_canonical_smiles"].tolist(), args.workers
    )):
        packed_mm[index] = fingerprint
        invalid += int(not ok)
        if index and index % 25000 == 0:
            print(json.dumps({"fingerprinted_candidates": index, "total": len(candidates)}), flush=True)
    packed_mm.flush()
    candidates[["spec_id", "candidate_hash", "candidate_rank_freq"]].to_csv(
        out_dir / "candidate_rows.tsv", sep="\t", index=False
    )
    pd.DataFrame({"spec_id": spec_ids}).to_csv(out_dir / "spectra.tsv", sep="\t", index=False)
    summary = {
        "schema": "ms2forge.ssr_validation_cache.v1",
        "status": "experimental_not_frozen",
        "source_split": "validation_only",
        "test_access": "closed",
        "n_candidates": int(len(candidates)),
        "n_spectra": int(len(spec_ids)),
        "invalid_candidate_fingerprints": invalid,
        "split_counts": split_counts,
        "spectra_audit": spectra_audit,
        "max_peaks": args.max_peaks,
        "candidate_table_sha256": sha256_file(Path(args.candidate_table)),
    }
    json_dump(summary, out_dir / "prepare_summary.json")
    (out_dir / "VALIDATION_CACHE_READY_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_model(embedding_dim: int = 256, dropout: float = 0.1):
    torch, nn, F = import_torch()

    class SpectrumStructureRanker(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = embedding_dim
            self.peak_mlp = nn.Sequential(
                nn.Linear(PEAK_DIM, hidden), nn.GELU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.peak_attention = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
            )
            self.meta_mlp = nn.Sequential(
                nn.Linear(META_DIM, hidden), nn.GELU(), nn.LayerNorm(hidden),
            )
            self.spectrum_head = nn.Sequential(
                nn.Linear(hidden * 3, hidden * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, embedding_dim),
            )
            self.molecule_head = nn.Sequential(
                nn.Linear(FP_BITS, 512), nn.GELU(), nn.LayerNorm(512),
                nn.Dropout(dropout), nn.Linear(512, embedding_dim),
            )

        def encode_spectrum(self, peaks, meta):
            h = self.peak_mlp(peaks)
            mask = peaks[..., 3] > 0
            attention = self.peak_attention(h).squeeze(-1).masked_fill(~mask, -1e4)
            attention = torch.softmax(attention, dim=1)
            pooled_attention = torch.sum(h * attention.unsqueeze(-1), dim=1)
            h_max = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
            h_max = torch.where(mask.any(dim=1, keepdim=True), h_max, torch.zeros_like(h_max))
            meta_h = self.meta_mlp(meta)
            return F.normalize(self.spectrum_head(
                torch.cat([pooled_attention, h_max, meta_h], dim=-1)
            ), dim=-1)

        def encode_molecule(self, fingerprint):
            return F.normalize(self.molecule_head(fingerprint), dim=-1)

        def forward(self, peaks, meta, chosen_fp, rejected_fp):
            spectrum = self.encode_spectrum(peaks, meta)
            chosen = self.encode_molecule(chosen_fp)
            rejected = self.encode_molecule(rejected_fp)
            return spectrum, chosen, rejected

    return SpectrumStructureRanker()


def unpack_fingerprints(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=FP_BITS, bitorder="big").astype(np.float32)


class TrainCacheDataset:
    def __init__(self, root: Path, indices: np.ndarray):
        self.root = root
        self.indices = np.asarray(indices, dtype=np.int64)
        self.spec_index = np.load(root / "pair_spec_index.npy", mmap_mode="r")
        self.peaks = np.load(root / "spectra_peaks.npy", mmap_mode="r")
        self.meta = np.load(root / "spectra_meta.npy", mmap_mode="r")
        self.chosen = np.load(root / "chosen_fp_packed.npy", mmap_mode="r")
        self.rejected = np.load(root / "rejected_fp_packed.npy", mmap_mode="r")
        self.chosen_group = np.load(root / "chosen_group.npy", mmap_mode="r")
        self.rejected_tanimoto = np.load(root / "rejected_tanimoto.npy", mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        spectrum_index = int(self.spec_index[index])
        return (
            np.asarray(self.peaks[spectrum_index], dtype=np.float32),
            np.asarray(self.meta[spectrum_index], dtype=np.float32),
            np.asarray(self.chosen[index], dtype=np.uint8),
            np.asarray(self.rejected[index], dtype=np.uint8),
            int(self.chosen_group[index]),
            float(self.rejected_tanimoto[index]),
        )


def collate_train(batch):
    torch, _, _ = import_torch()
    peaks, meta, chosen, rejected, group, rejected_tan = zip(*batch)
    return (
        torch.from_numpy(np.stack(peaks)),
        torch.from_numpy(np.stack(meta)),
        torch.from_numpy(unpack_fingerprints(np.stack(chosen))),
        torch.from_numpy(unpack_fingerprints(np.stack(rejected))),
        torch.as_tensor(group, dtype=torch.long),
        torch.as_tensor(rejected_tan, dtype=torch.float32),
    )


def multi_positive_nce(logits, groups):
    torch, _, _ = import_torch()
    positive = groups[:, None].eq(groups[None, :])
    numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def command_train(args: argparse.Namespace) -> None:
    torch, _, F = import_torch()
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    root = Path(args.train_cache).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((root / "prepare_summary.json").read_text())
    if prepare.get("source_split") != "train_only":
        raise RuntimeError("training cache is not audited train-only data")
    n_pairs = int(prepare["n_pairs"])
    rng = np.random.default_rng(args.seed)
    chosen_groups = np.load(root / "chosen_group.npy", mmap_mode="r")
    unique_groups = np.unique(chosen_groups)
    rng.shuffle(unique_groups)
    n_monitor_groups = max(1, int(round(len(unique_groups) * args.monitor_fraction)))
    monitor_groups = unique_groups[:n_monitor_groups]
    monitor_mask = np.isin(chosen_groups, monitor_groups)
    monitor_indices = np.flatnonzero(monitor_mask).astype(np.int64)
    train_indices = np.flatnonzero(~monitor_mask).astype(np.int64)
    if not len(train_indices) or not len(monitor_indices):
        raise RuntimeError("structure-disjoint internal monitor split is empty")
    train_set = TrainCacheDataset(root, train_indices)
    monitor_set = TrainCacheDataset(root, monitor_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_train,
        persistent_workers=args.workers > 0,
    )
    monitor_loader = DataLoader(
        monitor_set, batch_size=args.batch_size, shuffle=False,
        num_workers=max(0, min(args.workers, 2)), pin_memory=True, collate_fn=collate_train,
    )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(args.embedding_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def step_loss(batch, training: bool):
        peaks, meta, chosen_fp, rejected_fp, groups, rejected_tan = [x.to(device, non_blocking=True) for x in batch]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            spectrum, chosen, rejected = model(peaks, meta, chosen_fp, rejected_fp)
            pos = torch.sum(spectrum * chosen, dim=-1)
            neg = torch.sum(spectrum * rejected, dim=-1)
            pair_loss = F.softplus((args.margin - pos + neg) / args.pair_temperature).mean()
            logits = spectrum @ chosen.T / args.nce_temperature
            nce_loss = multi_positive_nce(logits, groups)
            pos_unit = (pos + 1.0) * 0.5
            neg_unit = (neg + 1.0) * 0.5
            regression = F.mse_loss(pos_unit, torch.ones_like(pos_unit)) + F.mse_loss(neg_unit, rejected_tan)
            loss = pair_loss + args.nce_weight * nce_loss + args.regression_weight * regression
        return loss, pair_loss.detach(), nce_loss.detach(), regression.detach(), (pos > neg).float().mean().detach()

    best_monitor = float("inf")
    history: list[dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(5, dtype=np.float64)
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, pair, nce, regression, pair_accuracy = step_loss(batch, True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            values = [loss.detach(), pair, nce, regression, pair_accuracy]
            batch_size = int(batch[0].shape[0])
            running += np.asarray([float(value.cpu()) for value in values]) * batch_size
            count += batch_size
        model.eval()
        monitor = np.zeros(5, dtype=np.float64)
        monitor_count = 0
        with torch.no_grad():
            for batch in monitor_loader:
                values = step_loss(batch, False)
                batch_size = int(batch[0].shape[0])
                monitor += np.asarray([float(value.cpu()) for value in values]) * batch_size
                monitor_count += batch_size
        train_values = running / max(count, 1)
        monitor_values = monitor / max(monitor_count, 1)
        record = {
            "epoch": epoch,
            "train_loss": train_values[0], "train_pair_accuracy": train_values[4],
            "monitor_loss": monitor_values[0], "monitor_pair_accuracy": monitor_values[4],
            "monitor_pair_loss": monitor_values[1], "monitor_nce_loss": monitor_values[2],
            "monitor_regression_loss": monitor_values[3], "elapsed_seconds": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": {"embedding_dim": args.embedding_dim, "dropout": args.dropout},
            "train_args": {k: v for k, v in vars(args).items() if k != "func"},
            "epoch": epoch, "history": history,
            "train_cache_summary": prepare,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if monitor_values[0] < best_monitor:
            best_monitor = float(monitor_values[0])
            torch.save(checkpoint, out_dir / "best.pt")
    summary = {
        "schema": "ms2forge.ssr_model.v1",
        "status": "experimental_not_frozen",
        "training_split": "train_only",
        "test_access": "closed",
        "device": str(device),
        "n_train_pairs": int(len(train_indices)),
        "n_internal_monitor_pairs": int(len(monitor_indices)),
        "n_train_truth_structures": int(len(unique_groups) - n_monitor_groups),
        "n_internal_monitor_truth_structures": int(n_monitor_groups),
        "internal_monitor_structure_disjoint": True,
        "best_internal_monitor_loss": best_monitor,
        "history": history,
        "checkpoint_sha256": sha256_file(out_dir / "best.pt"),
    }
    json_dump(summary, out_dir / "train_summary.json")
    (out_dir / "TRAINING_COMPLETE_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")


def command_score(args: argparse.Namespace) -> None:
    torch, _, _ = import_torch()
    cache = Path(args.candidate_cache).resolve()
    prepare = json.loads((cache / "prepare_summary.json").read_text())
    if prepare.get("source_split") != "validation_only":
        raise RuntimeError("scoring accepts only an audited validation cache; test is sealed")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = make_model(**model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    peaks = np.load(cache / "spectra_peaks.npy", mmap_mode="r")
    meta = np.load(cache / "spectra_meta.npy", mmap_mode="r")
    packed = np.load(cache / "candidate_fp_packed.npy", mmap_mode="r")
    spec_index = np.load(cache / "candidate_spec_index.npy", mmap_mode="r")
    rows = pd.read_csv(cache / "candidate_rows.tsv", sep="\t", dtype=str, keep_default_na=False)
    spectrum_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(peaks), args.batch_size):
            stop = min(start + args.batch_size, len(peaks))
            p = torch.from_numpy(np.array(peaks[start:stop], dtype=np.float32, copy=True)).to(device)
            m = torch.from_numpy(np.array(meta[start:stop], dtype=np.float32, copy=True)).to(device)
            spectrum_embeddings.append(model.encode_spectrum(p, m).cpu().numpy().astype(np.float32))
    spectrum_embeddings_array = np.concatenate(spectrum_embeddings, axis=0)
    scores = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            fp = torch.from_numpy(unpack_fingerprints(np.asarray(packed[start:stop]))).to(device)
            molecule = model.encode_molecule(fp).cpu().numpy()
            spectrum = spectrum_embeddings_array[np.asarray(spec_index[start:stop], dtype=np.int64)]
            scores[start:stop] = np.sum(spectrum * molecule, axis=1)
            if start and start % (args.batch_size * 50) == 0:
                print(json.dumps({"scored": start, "total": len(rows)}), flush=True)
    out_path = Path(args.out_scores).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "spec_id": rows["spec_id"], "candidate_hash": rows["candidate_hash"],
        "score": scores,
    }).to_csv(out_path, sep="\t", index=False, float_format="%.9g")
    summary = {
        "schema": "ms2forge.ssr_validation_scores.v1",
        "status": "experimental_not_frozen",
        "scored_split": "validation_only",
        "test_access": "closed",
        "n_rows": int(len(rows)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scores_sha256": sha256_file(out_path),
    }
    json_dump(summary, out_path.with_suffix(".json"))
    print(json.dumps(summary, indent=2), flush=True)


def within_group_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("spec_id", sort=False)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").fillna(0.0)
    return (frame[column] - mean) / std.where(std > 1e-12, 1.0)


def evaluate_rankings(frame: pd.DataFrame, score_column: str) -> dict:
    ordered = frame.sort_values(
        ["spec_id", score_column, "candidate_rank_freq_num", "candidate_hash"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    rank = ordered.groupby("spec_id", sort=False).cumcount() + 1
    ordered = ordered.assign(_rank=rank)
    top1 = ordered[ordered["_rank"] == 1]
    top10 = ordered[ordered["_rank"] <= 10]
    exact1 = int(top1["candidate_is_true"].sum())
    exact10 = int(top10.groupby("spec_id", sort=False)["candidate_is_true"].max().sum())
    tan1 = float(top1["candidate_tanimoto_to_truth"].mean())
    tan10 = float(top10.groupby("spec_id", sort=False)["candidate_tanimoto_to_truth"].max().mean())
    return {
        "n_spectra": int(frame["spec_id"].nunique()),
        "exact_top1_count": exact1,
        "exact_top1_accuracy": exact1 / frame["spec_id"].nunique(),
        "exact_top10_count": exact10,
        "exact_top10_accuracy": exact10 / frame["spec_id"].nunique(),
        "tanimoto_top1": tan1,
        "tanimoto_top10": tan10,
    }


def command_select(args: argparse.Namespace) -> None:
    rows = pd.read_csv(
        Path(args.candidate_cache) / "candidate_rows.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    base = pd.read_csv(args.base_scores, sep="\t", dtype={"spec_id": str, "candidate_hash": str})
    neural = pd.read_csv(args.neural_scores, sep="\t", dtype={"spec_id": str, "candidate_hash": str})
    targets = pd.read_csv(args.targets, sep="\t", dtype={"spec_id": str, "candidate_hash": str})
    frame = rows.merge(base.rename(columns={"score": "base_score"}), on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    frame = frame.merge(neural.rename(columns={"score": "neural_score"}), on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    frame = frame.merge(targets[["spec_id", "candidate_hash", "candidate_is_true", "candidate_tanimoto_to_truth"]], on=["spec_id", "candidate_hash"], how="left", validate="one_to_one")
    if frame[["base_score", "neural_score", "candidate_is_true", "candidate_tanimoto_to_truth"]].isna().any().any():
        raise RuntimeError("score/target join is incomplete")
    frame["candidate_rank_freq_num"] = pd.to_numeric(frame["candidate_rank_freq"], errors="coerce").fillna(10**9)
    frame["candidate_is_true"] = pd.to_numeric(frame["candidate_is_true"], errors="coerce").fillna(0).astype(int)
    frame["candidate_tanimoto_to_truth"] = pd.to_numeric(frame["candidate_tanimoto_to_truth"], errors="coerce").fillna(0.0)
    frame["base_z"] = within_group_zscore(frame, "base_score")
    frame["neural_z"] = within_group_zscore(frame, "neural_score")
    baseline = evaluate_rankings(frame, "base_score")
    grid: list[dict] = []
    for base_weight in [float(value) for value in args.base_weights.split(",")]:
        name = f"fused_{base_weight:.6f}"
        frame[name] = base_weight * frame["base_z"] + (1.0 - base_weight) * frame["neural_z"]
        metrics = evaluate_rankings(frame, name)
        metrics["base_weight"] = base_weight
        metrics["passes_exact_noninferiority"] = (
            metrics["exact_top1_count"] >= baseline["exact_top1_count"]
            and metrics["exact_top10_count"] >= baseline["exact_top10_count"]
        )
        grid.append(metrics)
        del frame[name]
    passing = [row for row in grid if row["passes_exact_noninferiority"]]
    selected = max(
        passing, key=lambda row: (row["tanimoto_top1"], row["tanimoto_top10"], row["base_weight"])
    ) if passing else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(out_dir / "validation_fusion_grid.tsv", sep="\t", index=False)
    summary = {
        "schema": "ms2forge.ssr_validation_selection.v1",
        "status": "experimental_not_frozen",
        "selection_split": "validation_only",
        "test_access": "closed",
        "guard": "exact Top-1 and Top-10 counts may not fall below the supplied base score",
        "baseline": baseline,
        "selected": selected,
        "grid": grid,
        "input_sha256": {
            "base_scores": sha256_file(Path(args.base_scores)),
            "neural_scores": sha256_file(Path(args.neural_scores)),
            "targets": sha256_file(Path(args.targets)),
        },
    }
    json_dump(summary, out_dir / "selection_summary.json")
    if selected is not None:
        weight = selected["base_weight"]
        frame["score"] = weight * frame["base_z"] + (1.0 - weight) * frame["neural_z"]
        frame[["spec_id", "candidate_hash", "score"]].to_csv(
            out_dir / "valid_selected_scores.tsv", sep="\t", index=False, float_format="%.9g"
        )
    (out_dir / "VALIDATION_SELECTION_COMPLETE_AT.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def command_select_winner(args: argparse.Namespace) -> None:
    models_root = Path(args.models_root).resolve()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    candidates: list[dict] = []
    baseline = None
    for variant in variants:
        summary_path = models_root / variant / "selection" / "selection_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing completed validation selection: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("selection_split") != "validation_only" or summary.get("test_access") != "closed":
            raise RuntimeError(f"invalid split boundary in {summary_path}")
        if baseline is None:
            baseline = summary["baseline"]
            candidates.append({
                "variant": "StageA-ER-SimRank-E8-A8",
                "base_weight": 1.0,
                "neural_weight": 0.0,
                **baseline,
            })
        selected = summary.get("selected")
        if selected is None or not selected.get("passes_exact_noninferiority", False):
            continue
        candidates.append({
            "variant": variant,
            "base_weight": float(selected["base_weight"]),
            "neural_weight": 1.0 - float(selected["base_weight"]),
            **{k: v for k, v in selected.items() if k != "base_weight"},
            "checkpoint_sha256": sha256_file(models_root / variant / "best.pt"),
            "selected_scores_sha256": sha256_file(
                models_root / variant / "selection" / "valid_selected_scores.tsv"
            ),
        })
    if baseline is None:
        raise RuntimeError("no validation summaries were loaded")
    winner = max(candidates, key=lambda row: (
        row["tanimoto_top1"], row["tanimoto_top10"],
        row["exact_top1_count"], row["exact_top10_count"],
        row["base_weight"], row["variant"] == "StageA-ER-SimRank-E8-A8",
    ))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ms2forge.ssr_provisional_validation_winner.v1",
        "status": "provisional_validation_winner_not_frozen",
        "selection_split": "validation_only",
        "test_access": "closed",
        "selection_rule": (
            "hard exact Top-1/Top-10 non-inferiority; maximize Tanimoto@1, "
            "then Tanimoto@10; then exact counts; prefer more Stage-A weight on ties"
        ),
        "multiple_comparison_warning": (
            "Three neural objectives and their fusion grids used the same validation split; "
            "this manifest is development evidence, not independent confirmation."
        ),
        "winner": winner,
        "candidates": candidates,
    }
    json_dump(payload, out_dir / "PROVISIONAL_VALIDATION_WINNER.json")
    pd.DataFrame(candidates).to_csv(
        out_dir / "provisional_validation_candidates.tsv", sep="\t", index=False
    )
    (out_dir / "PROVISIONAL_SELECTION_COMPLETE_AT.txt").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-train")
    p.add_argument("--pairs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split-table", required=True)
    p.add_argument("--spec-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--declared-split", default="train")
    p.add_argument("--max-peaks", type=int, default=128)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=command_prepare_train)

    p = sub.add_parser("prepare-candidates")
    p.add_argument("--candidate-table", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split-table", required=True)
    p.add_argument("--spec-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--declared-split", default="validation")
    p.add_argument("--max-peaks", type=int, default=128)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=command_prepare_candidates)

    p = sub.add_parser("train")
    p.add_argument("--train-cache", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--monitor-fraction", type=float, default=0.05)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--pair-temperature", type=float, default=0.1)
    p.add_argument("--nce-temperature", type=float, default=0.07)
    p.add_argument("--nce-weight", type=float, default=0.5)
    p.add_argument("--regression-weight", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.set_defaults(func=command_train)

    p = sub.add_parser("score")
    p.add_argument("--candidate-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-scores", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=2048)
    p.set_defaults(func=command_score)

    p = sub.add_parser("select")
    p.add_argument("--candidate-cache", required=True)
    p.add_argument("--base-scores", required=True)
    p.add_argument("--neural-scores", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-weights", default="0,0.25,0.5,0.65,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1")
    p.set_defaults(func=command_select)

    p = sub.add_parser("select-winner")
    p.add_argument("--models-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", default="PairOnly,Contrastive,ContrastiveTan")
    p.set_defaults(func=command_select_winner)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
