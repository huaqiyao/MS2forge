#!/usr/bin/env python3
"""Extract explicit peak-fragment evidence features for EvidenceRank-lite.

The extractor is label-free. It reads only candidate structures and raw MS/MS
peaks, builds constrained fragmentation hypotheses, creates mass-compatible
peak-fragment edges, and performs a deterministic one-to-one greedy assignment.
No truth label or split-level statistic is read, so the same frozen extractor
can be applied independently to train, valid, and test candidates.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276466621
SODIUM = 22.989218
POTASSIUM = 38.963158
AMMONIUM = 18.033823

COMMON_LOSSES: dict[str, tuple[float, dict[str, int]]] = {
    "H2O": (18.010565, {"H": 2, "O": 1}),
    "NH3": (17.026549, {"N": 1, "H": 3}),
    "CO": (27.994915, {"C": 1, "O": 1}),
    "CO2": (43.989830, {"C": 1, "O": 2}),
    "CH2O": (30.010565, {"C": 1, "H": 2, "O": 1}),
    "C2H4": (28.031300, {"C": 2, "H": 4}),
    "SO2": (63.961901, {"S": 1, "O": 2}),
}


@dataclass(frozen=True)
class FragmentHypothesis:
    neutral_mass: float
    prior: float
    source: str
    formula: str
    cut_bonds: tuple[int, ...]


@dataclass(frozen=True)
class IonHypothesis:
    mz: float
    prior: float
    ion_type: str
    source: str
    formula: str
    neutral_mass: float
    cut_bonds: tuple[int, ...]


@dataclass(frozen=True)
class CompatibleEdge:
    score: float
    peak_index: int
    hypothesis_index: int
    ppm_error: float
    normalized_error: float


FEATURE_COLUMNS = [
    "erl_available",
    "erl_num_observed_peaks",
    "erl_num_fragment_hypotheses",
    "erl_num_ion_hypotheses",
    "erl_num_compatible_edges",
    "erl_num_assigned_edges",
    "erl_assigned_peak_fraction",
    "erl_assigned_intensity_fraction",
    "erl_top10_assigned_intensity_fraction",
    "erl_top20_assigned_intensity_fraction",
    "erl_unexplained_top10_intensity_fraction",
    "erl_unexplained_top20_intensity_fraction",
    "erl_independent_intensity_fraction",
    "erl_assignment_conflict_intensity_loss",
    "erl_weighted_assignment_score",
    "erl_mean_abs_ppm",
    "erl_p90_abs_ppm",
    "erl_mean_matched_cleavage_prior",
    "erl_sum_matched_cleavage_prior",
    "erl_direct_assigned_edges",
    "erl_neutral_loss_assigned_edges",
    "erl_common_loss_assigned_edges",
    "erl_single_cut_assigned_edges",
    "erl_double_cut_assigned_edges",
    "erl_ring_open_assigned_edges",
    "erl_ambiguous_peak_fraction",
    "erl_mean_hypotheses_per_compatible_peak",
    "erl_peak_competition_penalty",
    "erl_mean_best_second_edge_margin",
    "erl_competing_hypothesis_fraction",
    "erl_fragment_hypothesis_density",
    "erl_evidence_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-structures", required=True, type=Path)
    parser.add_argument("--msg-root", required=True, type=Path)
    parser.add_argument("--out-table", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--evidence-jsonl", type=Path, default=None)
    parser.add_argument("--evidence-top-frequency-rank", type=int, default=3)
    parser.add_argument("--top-peaks", type=int, default=50)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--abs-tol", type=float, default=0.01)
    parser.add_argument("--max-single-cuts", type=int, default=64)
    parser.add_argument("--max-double-cut-pairs", type=int, default=32)
    parser.add_argument("--max-ring-open-pairs", type=int, default=24)
    parser.add_argument("--max-ion-hypotheses", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument(
        "--accept-all-valid-rows",
        action="store_true",
        help="Do not require included_reason=top_rank (useful for already finalized candidate tables).",
    )
    return parser.parse_args()


def fnum(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_ms_file(path: Path) -> tuple[float, str, tuple[tuple[float, float], ...]]:
    parent_mz = 0.0
    ionization = ""
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    if not path.exists():
        return parent_mz, ionization, tuple()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">parentmass"):
            parts = line.split()
            if len(parts) >= 2:
                parent_mz = fnum(parts[1], 0.0)
            continue
        if line.startswith(">ionization"):
            parts = line.split(maxsplit=1)
            ionization = parts[1] if len(parts) == 2 else ""
            continue
        if line.startswith(">ms2peaks"):
            in_peaks = True
            continue
        if line.startswith(">") and in_peaks:
            break
        if in_peaks:
            parts = line.split()
            if len(parts) >= 2:
                mz, intensity = fnum(parts[0], math.nan), fnum(parts[1], math.nan)
                if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                    peaks.append((mz, intensity))
    peaks.sort(key=lambda item: (-item[1], item[0]))
    return parent_mz, ionization, tuple(peaks)


@lru_cache(maxsize=250_000)
def cached_ms(msg_root: str, spec_id: str) -> tuple[float, str, tuple[tuple[float, float], ...]]:
    return parse_ms_file(Path(msg_root) / "spec_files" / f"{spec_id}.ms")


def adduct_shifts(ionization: str) -> tuple[tuple[str, float], ...]:
    text = ionization.upper().replace(" ", "")
    shifts: list[tuple[str, float]] = []
    if "+NA" in text:
        shifts.append(("sodiated", SODIUM))
    if "+K" in text:
        shifts.append(("potassiated", POTASSIUM))
    if "+NH4" in text:
        shifts.append(("ammoniated", AMMONIUM))
    if "-H" in text or text.endswith("-"):
        shifts.append(("deprotonated", -PROTON))
    if "+" in text:
        shifts.append(("protonated", PROTON))
    if not shifts:
        shifts.append(("protonated_default", PROTON))
    output: list[tuple[str, float]] = []
    for name, shift in shifts:
        if all(abs(shift - old_shift) > 1e-6 for _, old_shift in output):
            output.append((name, shift))
    return tuple(output)


def atom_counts(molecule: Chem.Mol) -> Counter[str]:
    counts: Counter[str] = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    counts["H"] += int(sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms()))
    return counts


def supports_loss(counts: Counter[str], required: dict[str, int]) -> bool:
    return all(counts[element] >= count for element, count in required.items())


def is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "C":
        return False
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == "O"
        for bond in atom.GetBonds()
    )


def cleavage_prior(bond: Chem.Bond) -> float:
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    if bond.GetIsAromatic():
        prior = 0.10
    elif bond.GetBondType() == Chem.BondType.SINGLE:
        prior = 0.75
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        prior = 0.20
    else:
        prior = 0.08
    if bond.IsInRing():
        prior *= 0.45
    if begin.GetSymbol() not in {"C", "H"} or end.GetSymbol() not in {"C", "H"}:
        prior *= 1.25
    if (is_carbonyl_carbon(begin) and end.GetSymbol() in {"N", "O", "S"}) or (
        is_carbonyl_carbon(end) and begin.GetSymbol() in {"N", "O", "S"}
    ):
        prior *= 1.35
    if min(begin.GetDegree(), end.GetDegree()) == 1:
        prior *= 0.90
    return min(max(prior, 0.03), 1.0)


def fragments_from_cuts(
    molecule: Chem.Mol,
    cut_bonds: tuple[int, ...],
) -> tuple[Chem.Mol, ...]:
    try:
        fragmented = Chem.FragmentOnBonds(molecule, list(cut_bonds), addDummies=False)
        fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    except Exception:
        return tuple()
    return tuple(fragments) if len(fragments) > 1 else tuple()


def fragment_record(
    fragment: Chem.Mol,
    parent_mass: float,
    prior: float,
    source: str,
    cut_bonds: tuple[int, ...],
) -> FragmentHypothesis | None:
    try:
        mass = float(rdMolDescriptors.CalcExactMolWt(fragment))
        formula = str(rdMolDescriptors.CalcMolFormula(fragment))
    except Exception:
        return None
    if mass <= 0 or abs(mass - parent_mass) <= 1e-6:
        return None
    return FragmentHypothesis(round(mass, 6), prior, source, formula, cut_bonds)


def ring_cut_pairs(molecule: Chem.Mol, limit: int) -> list[tuple[int, int]]:
    candidates: list[tuple[float, tuple[int, int]]] = []
    ring_info = molecule.GetRingInfo()
    for ring in ring_info.BondRings():
        for left, right in combinations(ring, 2):
            left_bond, right_bond = molecule.GetBondWithIdx(left), molecule.GetBondWithIdx(right)
            score = math.sqrt(cleavage_prior(left_bond) * cleavage_prior(right_bond))
            candidates.append((score, tuple(sorted((left, right)))))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    seen: set[tuple[int, int]] = set()
    output: list[tuple[int, int]] = []
    for _, pair in candidates:
        if pair in seen:
            continue
        seen.add(pair)
        output.append(pair)
        if len(output) >= limit:
            break
    return output


@lru_cache(maxsize=300_000)
def fragment_hypotheses(
    smiles: str,
    max_single_cuts: int,
    max_double_cut_pairs: int,
    max_ring_open_pairs: int,
) -> tuple[float, tuple[FragmentHypothesis, ...], tuple[tuple[str, float], ...]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 0.0, tuple(), tuple()
    try:
        parent_mass = float(rdMolDescriptors.CalcExactMolWt(molecule))
    except Exception:
        return 0.0, tuple(), tuple()

    nonring_bonds = [bond for bond in molecule.GetBonds() if not bond.IsInRing()]
    nonring_bonds.sort(key=lambda bond: (-cleavage_prior(bond), bond.GetIdx()))
    selected_single = nonring_bonds[:max_single_cuts]
    records: list[FragmentHypothesis] = []

    for bond in selected_single:
        prior = cleavage_prior(bond)
        cut = (bond.GetIdx(),)
        for fragment in fragments_from_cuts(molecule, cut):
            record = fragment_record(fragment, parent_mass, prior, "single_cut", cut)
            if record is not None:
                records.append(record)

    double_candidates: list[tuple[float, tuple[int, int]]] = []
    for left, right in combinations(selected_single[:24], 2):
        pair = tuple(sorted((left.GetIdx(), right.GetIdx())))
        prior = math.sqrt(cleavage_prior(left) * cleavage_prior(right)) * 0.75
        double_candidates.append((prior, pair))
    double_candidates.sort(key=lambda item: (-item[0], item[1]))
    for prior, cut in double_candidates[:max_double_cut_pairs]:
        for fragment in fragments_from_cuts(molecule, cut):
            record = fragment_record(fragment, parent_mass, prior, "double_cut", cut)
            if record is not None:
                records.append(record)

    for cut in ring_cut_pairs(molecule, max_ring_open_pairs):
        left, right = (molecule.GetBondWithIdx(index) for index in cut)
        prior = math.sqrt(cleavage_prior(left) * cleavage_prior(right)) * 0.70
        for fragment in fragments_from_cuts(molecule, cut):
            record = fragment_record(fragment, parent_mass, prior, "ring_open", cut)
            if record is not None:
                records.append(record)

    # Deduplicate nearly identical fragment masses while retaining the strongest
    # mechanistic hypothesis. This prevents molecular complexity from inflating
    # evidence solely by generating redundant fragments.
    best_by_mass: dict[float, FragmentHypothesis] = {}
    for record in records:
        key = round(record.neutral_mass, 5)
        old = best_by_mass.get(key)
        if old is None or (record.prior, record.source, record.cut_bonds) > (old.prior, old.source, old.cut_bonds):
            best_by_mass[key] = record

    counts = atom_counts(molecule)
    supported_losses = tuple(
        (name, mass)
        for name, (mass, required) in COMMON_LOSSES.items()
        if supports_loss(counts, required)
    )
    ordered = tuple(sorted(best_by_mass.values(), key=lambda item: (item.neutral_mass, -item.prior, item.source)))
    return parent_mass, ordered, supported_losses


def ion_hypotheses(
    parent_mz: float,
    ionization: str,
    fragments: tuple[FragmentHypothesis, ...],
    supported_losses: tuple[tuple[str, float], ...],
    max_hypotheses: int,
) -> tuple[IonHypothesis, ...]:
    hypotheses: list[IonHypothesis] = []
    for fragment in fragments:
        for adduct_name, shift in adduct_shifts(ionization):
            mz = fragment.neutral_mass + shift
            if mz > 0:
                hypotheses.append(
                    IonHypothesis(
                        round(mz, 6),
                        fragment.prior,
                        f"direct:{adduct_name}",
                        fragment.source,
                        fragment.formula,
                        fragment.neutral_mass,
                        fragment.cut_bonds,
                    )
                )
        neutral_loss_mz = parent_mz - fragment.neutral_mass
        if neutral_loss_mz > 0:
            hypotheses.append(
                IonHypothesis(
                    round(neutral_loss_mz, 6),
                    fragment.prior * 0.70,
                    "neutral_loss:fragment",
                    fragment.source,
                    fragment.formula,
                    fragment.neutral_mass,
                    fragment.cut_bonds,
                )
            )
    for name, loss_mass in supported_losses:
        mz = parent_mz - loss_mass
        if mz > 0:
            hypotheses.append(
                IonHypothesis(
                    round(mz, 6),
                    0.45,
                    f"neutral_loss:{name}",
                    "common_loss",
                    name,
                    loss_mass,
                    tuple(),
                )
            )

    best_by_key: dict[tuple[float, str], IonHypothesis] = {}
    for hypothesis in hypotheses:
        ion_class = "direct" if hypothesis.ion_type.startswith("direct") else "neutral_loss"
        key = (round(hypothesis.mz, 5), ion_class)
        old = best_by_key.get(key)
        if old is None or hypothesis.prior > old.prior:
            best_by_key[key] = hypothesis
    ordered = sorted(best_by_key.values(), key=lambda item: (-item.prior, item.mz, item.ion_type))
    ordered = ordered[:max_hypotheses]
    return tuple(sorted(ordered, key=lambda item: (item.mz, -item.prior, item.ion_type)))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(math.ceil(fraction * len(ordered))) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def compatible_edges(
    peaks: tuple[tuple[float, float], ...],
    hypotheses: tuple[IonHypothesis, ...],
    ppm: float,
    abs_tol: float,
) -> tuple[list[CompatibleEdge], list[int], list[int]]:
    mzs = [hypothesis.mz for hypothesis in hypotheses]
    peak_degrees = [0] * len(peaks)
    hypothesis_degrees = [0] * len(hypotheses)
    edges: list[CompatibleEdge] = []
    max_intensity = max((intensity for _, intensity in peaks), default=1.0)
    for peak_index, (peak_mz, intensity) in enumerate(peaks):
        tolerance = max(abs_tol, ppm * peak_mz * 1e-6)
        left = bisect.bisect_left(mzs, peak_mz - tolerance)
        right = bisect.bisect_right(mzs, peak_mz + tolerance)
        intensity_weight = 0.25 + 0.75 * math.sqrt(intensity / max_intensity)
        for hypothesis_index in range(left, right):
            hypothesis = hypotheses[hypothesis_index]
            error = abs(peak_mz - hypothesis.mz)
            normalized_error = error / max(tolerance, 1e-12)
            mass_score = math.exp(-0.5 * normalized_error * normalized_error)
            edge_score = mass_score * intensity_weight * hypothesis.prior
            ppm_error = error / max(hypothesis.mz, 1e-12) * 1e6
            edges.append(
                CompatibleEdge(edge_score, peak_index, hypothesis_index, ppm_error, normalized_error)
            )
            peak_degrees[peak_index] += 1
            hypothesis_degrees[hypothesis_index] += 1
    edges.sort(key=lambda edge: (-edge.score, edge.normalized_error, edge.peak_index, edge.hypothesis_index))
    return edges, peak_degrees, hypothesis_degrees


def assign_edges(edges: list[CompatibleEdge]) -> list[CompatibleEdge]:
    assigned_peaks: set[int] = set()
    assigned_hypotheses: set[int] = set()
    assignment: list[CompatibleEdge] = []
    for edge in edges:
        if edge.peak_index in assigned_peaks or edge.hypothesis_index in assigned_hypotheses:
            continue
        assigned_peaks.add(edge.peak_index)
        assigned_hypotheses.add(edge.hypothesis_index)
        assignment.append(edge)
    assignment.sort(key=lambda edge: edge.peak_index)
    return assignment


def evidence_features(
    spec_id: str,
    smiles: str,
    msg_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    parent_mz, ionization, all_peaks = cached_ms(str(msg_root), spec_id)
    peaks = tuple(all_peaks[: args.top_peaks])
    parent_mass, fragments, supported_losses = fragment_hypotheses(
        smiles,
        args.max_single_cuts,
        args.max_double_cut_pairs,
        args.max_ring_open_pairs,
    )
    hypotheses = ion_hypotheses(
        parent_mz,
        ionization,
        fragments,
        supported_losses,
        args.max_ion_hypotheses,
    )
    edges, peak_degrees, hypothesis_degrees = compatible_edges(peaks, hypotheses, args.ppm, args.abs_tol)
    assignment = assign_edges(edges)

    total_intensity = sum(intensity for _, intensity in peaks) or 1.0
    top10_total = sum(intensity for _, intensity in peaks[:10]) or 1.0
    top20_total = sum(intensity for _, intensity in peaks[:20]) or 1.0
    assigned_peak_indices = {edge.peak_index for edge in assignment}
    compatible_peak_indices = {edge.peak_index for edge in edges}
    assigned_intensity = sum(peaks[index][1] for index in assigned_peak_indices)
    assigned_top10 = sum(peaks[index][1] for index in assigned_peak_indices if index < 10)
    assigned_top20 = sum(peaks[index][1] for index in assigned_peak_indices if index < 20)
    independent_intensity = sum(peaks[index][1] for index in compatible_peak_indices)
    ppm_errors = [edge.ppm_error for edge in assignment]
    assigned_hypotheses = [hypotheses[edge.hypothesis_index] for edge in assignment]
    compatible_degrees = [degree for degree in peak_degrees if degree > 0]
    competing_hypotheses = sum(degree > 1 for degree in hypothesis_degrees)
    edge_scores_by_peak: dict[int, list[float]] = {}
    for edge in edges:
        edge_scores_by_peak.setdefault(edge.peak_index, []).append(edge.score)
    competition_penalty = 0.0
    best_second_margins: list[float] = []
    for peak_index, edge_scores in edge_scores_by_peak.items():
        ordered_scores = sorted(edge_scores, reverse=True)
        best = ordered_scores[0]
        total = sum(ordered_scores)
        if len(ordered_scores) > 1:
            second = ordered_scores[1]
            best_second_margins.append((best - second) / max(best, 1e-12))
            competition_penalty += peaks[peak_index][1] * (1.0 - best / max(total, 1e-12))

    direct_n = sum(hypothesis.ion_type.startswith("direct") for hypothesis in assigned_hypotheses)
    neutral_loss_n = sum(hypothesis.ion_type.startswith("neutral_loss") for hypothesis in assigned_hypotheses)
    common_loss_n = sum(hypothesis.source == "common_loss" for hypothesis in assigned_hypotheses)
    single_n = sum(hypothesis.source == "single_cut" for hypothesis in assigned_hypotheses)
    double_n = sum(hypothesis.source == "double_cut" for hypothesis in assigned_hypotheses)
    ring_n = sum(hypothesis.source == "ring_open" for hypothesis in assigned_hypotheses)
    mean_prior = (
        sum(hypothesis.prior for hypothesis in assigned_hypotheses) / len(assigned_hypotheses)
        if assigned_hypotheses
        else 0.0
    )
    sum_prior = sum(hypothesis.prior for hypothesis in assigned_hypotheses)
    assigned_fraction = assigned_intensity / total_intensity
    conflict_loss = max(0.0, independent_intensity - assigned_intensity) / total_intensity
    unexplained_top10 = max(0.0, 1.0 - assigned_top10 / top10_total)
    unexplained_top20 = max(0.0, 1.0 - assigned_top20 / top20_total)
    weighted_score = sum(edge.score for edge in assignment) / max(len(peaks), 1)
    evidence_score = (
        1.5 * assigned_fraction
        + 0.8 * (assigned_top20 / top20_total)
        + 0.5 * weighted_score
        + 0.2 * mean_prior
        - 0.8 * conflict_loss
        - 0.6 * (competition_penalty / total_intensity)
        - 0.5 * unexplained_top10
    )

    parsed_molecule = Chem.MolFromSmiles(smiles)
    bond_count = parsed_molecule.GetNumBonds() if parsed_molecule is not None else 0
    features: dict[str, float | int] = {
        "erl_available": int(bool(peaks) and bool(hypotheses)),
        "erl_num_observed_peaks": len(peaks),
        "erl_num_fragment_hypotheses": len(fragments),
        "erl_num_ion_hypotheses": len(hypotheses),
        "erl_num_compatible_edges": len(edges),
        "erl_num_assigned_edges": len(assignment),
        "erl_assigned_peak_fraction": len(assigned_peak_indices) / max(len(peaks), 1),
        "erl_assigned_intensity_fraction": assigned_fraction,
        "erl_top10_assigned_intensity_fraction": assigned_top10 / top10_total,
        "erl_top20_assigned_intensity_fraction": assigned_top20 / top20_total,
        "erl_unexplained_top10_intensity_fraction": unexplained_top10,
        "erl_unexplained_top20_intensity_fraction": unexplained_top20,
        "erl_independent_intensity_fraction": independent_intensity / total_intensity,
        "erl_assignment_conflict_intensity_loss": conflict_loss,
        "erl_weighted_assignment_score": weighted_score,
        "erl_mean_abs_ppm": sum(ppm_errors) / len(ppm_errors) if ppm_errors else 0.0,
        "erl_p90_abs_ppm": percentile(ppm_errors, 0.90),
        "erl_mean_matched_cleavage_prior": mean_prior,
        "erl_sum_matched_cleavage_prior": sum_prior,
        "erl_direct_assigned_edges": direct_n,
        "erl_neutral_loss_assigned_edges": neutral_loss_n,
        "erl_common_loss_assigned_edges": common_loss_n,
        "erl_single_cut_assigned_edges": single_n,
        "erl_double_cut_assigned_edges": double_n,
        "erl_ring_open_assigned_edges": ring_n,
        "erl_ambiguous_peak_fraction": sum(degree > 1 for degree in peak_degrees) / max(len(peaks), 1),
        "erl_mean_hypotheses_per_compatible_peak": sum(compatible_degrees) / max(len(compatible_degrees), 1),
        "erl_peak_competition_penalty": competition_penalty / total_intensity,
        "erl_mean_best_second_edge_margin": sum(best_second_margins) / max(len(best_second_margins), 1),
        "erl_competing_hypothesis_fraction": competing_hypotheses / max(len(hypotheses), 1),
        "erl_fragment_hypothesis_density": len(fragments) / max(bond_count, 1),
        "erl_evidence_score": evidence_score,
    }

    map_edges = []
    for edge in assignment:
        peak_mz, intensity = peaks[edge.peak_index]
        hypothesis = hypotheses[edge.hypothesis_index]
        map_edges.append(
            {
                "peak_rank_by_intensity": edge.peak_index + 1,
                "observed_mz": peak_mz,
                "intensity": intensity,
                "hypothesis_mz": hypothesis.mz,
                "ppm_error": edge.ppm_error,
                "edge_score": edge.score,
                "ion_type": hypothesis.ion_type,
                "fragment_source": hypothesis.source,
                "fragment_formula": hypothesis.formula,
                "fragment_neutral_mass": hypothesis.neutral_mass,
                "cleavage_prior": hypothesis.prior,
                "cut_bonds": list(hypothesis.cut_bonds),
            }
        )
    evidence_map = {
        "spec_id": spec_id,
        "candidate_smiles": smiles,
        "parent_mz": parent_mz,
        "candidate_exact_mass": parent_mass,
        "ionization": ionization,
        "features": features,
        "assigned_edges": map_edges,
    }
    return features, evidence_map


def selected_rows(path: Path, args: argparse.Namespace) -> Iterable[dict[str, str]]:
    emitted = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row_index % args.num_shards != args.shard_index:
                continue
            if not args.accept_all_valid_rows and row.get("included_reason") not in {"top_rank", ""}:
                continue
            if "rdkit_valid" in row and not truthy(row.get("rdkit_valid")):
                continue
            yield row
            emitted += 1
            if args.max_rows is not None and emitted >= args.max_rows:
                break


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    args.candidate_structures = args.candidate_structures.expanduser().resolve()
    args.msg_root = args.msg_root.expanduser().resolve()
    args.out_table = args.out_table.expanduser().resolve()
    args.summary_json = args.summary_json.expanduser().resolve()
    if args.evidence_jsonl:
        args.evidence_jsonl = args.evidence_jsonl.expanduser().resolve()
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.evidence_jsonl:
        args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)

    temp_table = args.out_table.with_suffix(args.out_table.suffix + ".tmp")
    evidence_temp = args.evidence_jsonl.with_suffix(args.evidence_jsonl.suffix + ".tmp") if args.evidence_jsonl else None
    processed = available = evidence_maps = 0
    feature_sums = Counter()
    feature_sums_sq = Counter()

    evidence_handle = evidence_temp.open("w", encoding="utf-8") if evidence_temp else None
    try:
        with temp_table.open("w", newline="", encoding="utf-8") as output_handle:
            fieldnames = [
                "spec_id",
                "candidate_hash",
                "candidate_canonical_smiles",
                "candidate_rank_freq",
                *FEATURE_COLUMNS,
            ]
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in selected_rows(args.candidate_structures, args):
                spec_id = str(row.get("spec_id", "")).strip()
                candidate_hash = str(row.get("candidate_hash", "")).strip()
                smiles = (
                    str(row.get("candidate_canonical_smiles", "")).strip()
                    or str(row.get("candidate_canonical_smiles_nostereo", "")).strip()
                    or str(row.get("smiles", "")).strip()
                )
                if not spec_id or not candidate_hash or not smiles:
                    continue
                features, evidence_map = evidence_features(spec_id, smiles, args.msg_root, args)
                output = {
                    "spec_id": spec_id,
                    "candidate_hash": candidate_hash,
                    "candidate_canonical_smiles": smiles,
                    "candidate_rank_freq": row.get("candidate_rank_freq", ""),
                    **features,
                }
                writer.writerow(output)
                processed += 1
                available += int(features["erl_available"])
                for feature in FEATURE_COLUMNS:
                    value = fnum(features.get(feature), 0.0)
                    feature_sums[feature] += value
                    feature_sums_sq[feature] += value * value
                if evidence_handle is not None and int(fnum(row.get("candidate_rank_freq"), 10**9)) <= args.evidence_top_frequency_rank:
                    evidence_map["candidate_hash"] = candidate_hash
                    evidence_map["candidate_rank_freq"] = int(fnum(row.get("candidate_rank_freq"), 10**9))
                    evidence_handle.write(json.dumps(evidence_map, ensure_ascii=False) + "\n")
                    evidence_maps += 1
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"[EvidenceRank-lite] rows={processed} available={available}", flush=True)
    finally:
        if evidence_handle is not None:
            evidence_handle.close()

    os.replace(temp_table, args.out_table)
    if args.evidence_jsonl and evidence_temp:
        os.replace(evidence_temp, args.evidence_jsonl)

    feature_moments = {}
    for feature in FEATURE_COLUMNS:
        mean = feature_sums[feature] / max(processed, 1)
        variance = max(0.0, feature_sums_sq[feature] / max(processed, 1) - mean * mean)
        feature_moments[feature] = {"mean": mean, "sd": math.sqrt(variance)}
    summary = {
        "extractor": "MS2Forge EvidenceRank-lite v0.2",
        "policy": "label-free candidate-spectrum evidence extraction; no split-level fitting",
        "candidate_structures": str(args.candidate_structures),
        "candidate_structures_sha256": sha256(args.candidate_structures),
        "msg_root": str(args.msg_root),
        "out_table": str(args.out_table),
        "out_table_sha256": sha256(args.out_table),
        "evidence_jsonl": str(args.evidence_jsonl) if args.evidence_jsonl else None,
        "processed_rows": processed,
        "available_rows": available,
        "evidence_maps": evidence_maps,
        "parameters": {
            "top_peaks": args.top_peaks,
            "ppm": args.ppm,
            "abs_tol": args.abs_tol,
            "max_single_cuts": args.max_single_cuts,
            "max_double_cut_pairs": args.max_double_cut_pairs,
            "max_ring_open_pairs": args.max_ring_open_pairs,
            "max_ion_hypotheses": args.max_ion_hypotheses,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "max_rows": args.max_rows,
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_moments_for_diagnostics_only": feature_moments,
        "important_boundary": "Do not use valid/test feature moments for normalization; fit normalization on train only.",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
