"""Deterministic fragment-v2 features and train-fitted proxy statistics."""
import bisect
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rerank.spectra import read_spectrum

LOSSES = dict(H2O=18.010565, NH3=17.026549, CO=27.994915, CO2=43.989830,
              CH2O=30.010565, C2H4=28.031300, SO2=63.961901)
WEIGHTS = dict(mass_frag_v2_top_intensity_explained=.8, mass_frag_v2_parent_mass_score=.5,
               mass_frag_v2_common_loss_hits=.05, mass_frag_v2_common_loss_weighted_hits=.05,
               mass_frag_v2_candidate_common_loss_support=.15,
               mass_frag_v2_candidate_common_loss_observed=.35, mass_frag_v2_fragment_density=.1)


def adduct_shifts(ion):
    text = ion.upper().replace(' ', '')
    shifts = []
    for marker, mass in [('+NA', 22.989218), ('+K', 38.963158), ('+NH4', 18.033823)]:
        if marker in text:
            shifts.append(mass)
    if '-H' in text or text.endswith('-'):
        shifts.append(-1.007276466621)
    if '+' in text or not shifts:
        shifts.append(1.007276466621)
    return shifts


@lru_cache(maxsize=200000)
def fragment_masses(smiles, max_bonds=128):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid candidate: {smiles}')
    parent = float(rdMolDescriptors.CalcExactMolWt(mol))
    masses = set()
    for bond in list(mol.GetBonds())[:max_bonds]:
        if bond.IsInRing():
            continue
        try:
            cut = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
            fragments = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        if len(fragments) <= 1:
            continue
        for fragment in fragments:
            try:
                mass = float(rdMolDescriptors.CalcExactMolWt(fragment))
            except Exception:
                continue
            if mass > 0 and abs(mass - parent) > 1e-6:
                masses.add(round(mass, 6))
    return parent, tuple(sorted(masses)), mol.GetNumBonds()


def match(values, target, tolerance):
    index = bisect.bisect_left(values, target - tolerance)
    return index < len(values) and abs(values[index] - target) <= tolerance


@lru_cache(maxsize=8192)
def spectrum(path):
    return read_spectrum(path)


def mass_row(item):
    sid, smiles, root = item
    parent_mz, ion, peaks = spectrum(str(Path(root) / 'spec_files' / f'{sid}.ms'))
    parent, fragments, n_bonds = fragment_masses(smiles)
    shifts = adduct_shifts(ion)
    direct = sorted({round(m + shift, 6) for m in fragments for shift in shifts if m + shift > 0})
    neutral = sorted({round(parent_mz - m, 6) for m in fragments if parent_mz > m})
    losses = sorted({round(parent - m, 6) for m in fragments if parent > m})
    top = peaks[:20]
    explained = sum(i for mz, i in top if match(direct, mz, max(.01, mz * 20e-6))
                    or match(neutral, mz, max(.01, mz * 20e-6))) / (sum(i for _, i in top) or 1)
    error = min(abs(parent_mz - (parent + shift)) / (parent + shift) * 1e6
                if parent + shift > 0 else 1e9 for shift in shifts)
    row = dict(mass_frag_v2_available=1, mass_frag_v2_top_intensity_explained=explained,
               mass_frag_v2_parent_ppm_error=error,
               mass_frag_v2_parent_mass_score=math.exp(-min(error, 200) / 25),
               mass_frag_v2_fragment_density=len(fragments) / max(n_bonds, 1))
    support = observed = hits = 0
    weighted = 0.0
    for name, loss in LOSSES.items():
        supported = match(losses, loss, max(.01, loss * 20e-6))
        target = parent_mz - loss
        tol = max(.01, target * 20e-6)
        err = min(abs(mz - target) for mz, _ in peaks)
        hit = target > 0 and err <= tol
        row[f'mass_frag_v2_candidate_loss_{name}_support'] = int(supported)
        row[f'mass_frag_v2_candidate_loss_{name}_observed'] = int(supported and hit)
        row[f'mass_frag_v2_loss_{name}'] = int(hit)
        support += supported
        observed += supported and hit
        hits += hit
        weighted += 1 / (1 + err / tol) if hit else 0
    row.update(mass_frag_v2_candidate_common_loss_support=int(support),
               mass_frag_v2_candidate_common_loss_observed=int(observed),
               mass_frag_v2_common_loss_hits=int(hits), mass_frag_v2_common_loss_weighted_hits=weighted)
    return row


def fit_stats(frame):
    stats = {}
    for name in WEIGHTS:
        values = frame[name].to_numpy(float)
        variance = float(np.var(values))
        stats[name] = {'mean': float(np.mean(values)), 'sd': math.sqrt(variance) if variance > 1e-12 else 1.0}
    return {'schema': 'ms2forge.fragment_v2_stats.v1', 'source_split': 'train',
            'feature_standardization': stats, 'weights': WEIGHTS}


def proxy_scores(frame, payload):
    stats = payload['feature_standardization']
    result = np.zeros(len(frame), dtype=float)
    for name, weight in WEIGHTS.items():
        mean, sd = float(stats[name]['mean']), float(stats[name]['sd'])
        if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise ValueError(f'Invalid proxy statistics: {name}')
        result += weight * (frame[name].to_numpy(float) - mean) / sd
    return result
