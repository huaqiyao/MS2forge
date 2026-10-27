"""Spectrum and graph input adapters for the MSG/NPLIB1 encoder."""
import csv
import json
import re
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from utils.formula import SPECTRUM_PRECURSOR_INDEX
from utils.collision_energy import collision_energy_bin

ELEMENTS = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I')
ATOMS = {s: i for i, s in enumerate(('B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'))}
BONDS = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
         Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


def table(filename):
    with Path(filename).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter=',' if Path(filename).suffix == '.csv' else '\t'))


def metadata(root, supervised=False):
    """Spectrum export never consumes SMILES or reference graph fields."""
    root = Path(root)
    split = {}
    for row in table(root / 'split.tsv'):
        sid = row['name']
        if sid in split:
            raise ValueError(f'Duplicate split ID: {sid}')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Unsupported split: {row["split"]}')
        split[sid] = row['split']
    rows, seen = [], set()
    for raw in table(root / 'labels.tsv'):
        sid = raw.get('spec') or raw.get('identifier') or raw.get('sample_id')
        if sid not in split or (supervised and split[sid] == 'test'):
            continue
        if sid in seen or Path(sid).name != sid or '/' in sid or '\\' in sid:
            raise ValueError(f'Duplicate or unsafe spectrum ID: {sid}')
        seen.add(sid)
        row = dict(spec=sid, split=split[sid], formula=raw.get('formula', ''),
                   ionization=raw.get('ionization') or raw.get('adduct', ''),
                    instrument=raw.get('instrument', 'NONE'),
                    energy_bin=collision_energy_bin(raw, root, sid))
        if supervised:
            row['smiles'] = raw['smiles']
        rows.append(row)
    wanted = {k for k, v in split.items() if not supervised or v != 'test'}
    if seen != wanted:
        raise ValueError(f'Missing label rows for {len(wanted - seen)} spectrum IDs')
    return rows


def spectrum_counts(root, sid, adduct):
    if adduct not in SPECTRUM_PRECURSOR_INDEX:
        raise ValueError(f'{sid}: unsupported encoder adduct {adduct}')
    source = Path(root) / 'subformulae/default_subformulae' / (sid + '.json')
    # These are observed-spectrum fragment annotations, not inferred from truth SMILES.
    obj = json.loads(source.read_text(encoding='utf-8'))
    forms = ((obj.get('output_tbl') or {}).get('formula') or [])[:128]
    values = np.zeros((128, 9), dtype=np.int16)
    for j, form in enumerate(forms):
        parts = re.findall(r'([A-Z][a-z]?)(\d*)', form)
        if ''.join(a+b for a,b in parts) != form:
            raise ValueError(f'{sid}: malformed fragment formula {form}')
        counts = {}
        for element, count in parts:
            counts[element] = counts.get(element, 0) + (int(count) if count else 1)
        # Match the source adapter's nine-element vocabulary and clipping.
        values[j] = [min(counts.get(e, 0), 149) for e in ELEMENTS]
    return values, len(forms), SPECTRUM_PRECURSOR_INDEX[adduct]


def spectrum_batch(rows, root, device):
    arrays = [spectrum_counts(root, r['spec'], r['ionization']) for r in rows]
    counts = torch.tensor(np.stack([a[0] for a in arrays]), dtype=torch.long)
    peaks = torch.tensor([a[1] for a in arrays])
    adduct = torch.tensor([a[2] for a in arrays])
    div = torch.exp(torch.arange(0, 16, 2).float() * -(torch.log(torch.tensor(10000.)) / 8))
    position = torch.arange(150).float()[:, None]
    lut = torch.zeros(150, 16)
    lut[:, 0::2], lut[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
    valid = torch.arange(128)[None, :] < peaks[:, None]
    form = lut[counts].reshape(len(rows), 128, 144) * valid[:, :, None]
    mask = torch.cat([torch.zeros(len(rows), 1, dtype=torch.bool), ~valid], 1)
    sos = torch.zeros(len(rows), 1, 13)
    sos[torch.arange(len(rows)), 0, adduct] = 1
    energy = torch.tensor([int(r['energy_bin']) if 'energy_bin' in r else
                           collision_energy_bin(r, root, r['spec']) for r in rows])
    if ((energy < 0) | (energy > 10)).any():
        raise ValueError('Collision-energy bins must be in [0, 10]')
    sos[torch.arange(len(rows)), 0, 2 + energy] = 1
    return sos.to(device), form.to(device), mask.to(device)


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    mol = Chem.MolFromSmiles(key)
    if len(Chem.GetMolFrags(mol)) != 1 or not mol.GetNumBonds() or mol.GetNumAtoms() > 150:
        raise ValueError(f'Encoder requires a connected bonded molecule with <=150 atoms: {smiles}')
    if any(a.GetSymbol() not in ATOMS or a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError(f'Unsupported encoder atom/isotope: {smiles}')
    nodes = np.array([ATOMS[a.GetSymbol()] for a in mol.GetAtoms()], dtype=np.int8)
    edges = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[i, j] = edges[j, i] = BONDS[bond.GetBondType()]
    return key, nodes, edges


def graph_batch(smiles, device):
    graphs = [graph(s) for s in smiles]
    n = max(len(g[1]) for g in graphs)
    nodes = torch.zeros(len(graphs), n, dtype=torch.long)
    edges = torch.zeros(len(graphs), n, n, dtype=torch.long)
    mask = torch.zeros(len(graphs), n, dtype=torch.bool)
    for i, (_, x, e) in enumerate(graphs):
        size = len(x)
        nodes[i, :size] = torch.from_numpy(x.astype(np.int64))
        edges[i, :size, :size] = torch.from_numpy(e.astype(np.int64))
        mask[i, :size] = True
    X = torch.nn.functional.one_hot(nodes, 11).float() * mask[:, :, None]
    E = torch.nn.functional.one_hot(edges, 5).float() * mask[:, :, None, None] * mask[:, None, :, None]
    return X.to(device), E.to(device), torch.ones(len(graphs), 1, device=device), mask.to(device)
