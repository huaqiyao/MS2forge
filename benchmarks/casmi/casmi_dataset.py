#!/usr/bin/env python3
"""CASMI dataset adapter for formula-predicted end-to-end structure generation."""

from __future__ import annotations

import re

import torch

from utils.dataset import DiffMSMSGDataset


DIFFMS_ATOM_TYPES = {
    "B": 0, "C": 1, "N": 2, "O": 3, "F": 4, "Si": 5,
    "P": 6, "S": 7, "Cl": 8, "Br": 9, "I": 10,
}


def formula_atom_types(formula: str) -> list[int]:
    """Expand a neutral molecular formula into DiffMS heavy-atom type indices."""
    atom_types: list[int] = []
    reconstructed = ""
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", str(formula)):
        symbol, count_text = match.groups()
        reconstructed += match.group(0)
        count = int(count_text) if count_text else 1
        if symbol == "H":
            continue
        if symbol not in DIFFMS_ATOM_TYPES:
            raise ValueError(f"Unsupported generation-formula element {symbol!r} in {formula!r}")
        atom_types.extend([DIFFMS_ATOM_TYPES[symbol]] * count)
    if reconstructed != str(formula):
        raise ValueError(f"Could not fully parse generation formula {formula!r}")
    if not atom_types:
        raise ValueError(f"Generation formula has no heavy atoms: {formula!r}")
    return atom_types


class CASMIDiffMSDataset(DiffMSMSGDataset):
    """Use ``generation_formula`` for fixed node composition while retaining true SMILES.

    Oracle labels omit ``generation_formula`` and therefore use the original validated
    true-structure graph path. Predicted-formula labels include it and get a formula-only
    graph: exact heavy-atom multiset, no truth bonds. The standard DiffMS collator then
    constructs the complete half-edge set, so no true connectivity enters generation.
    """

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        row = self.labels_df.iloc[idx]
        if "generation_formula" not in self.labels_df.columns:
            return data
        value = row.get("generation_formula")
        if value is None or str(value).strip() in {"", "nan"}:
            return data
        atom_types = formula_atom_types(str(value).strip())
        data.x = torch.nn.functional.one_hot(
            torch.tensor(atom_types, dtype=torch.long), num_classes=11
        ).float()
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        data.num_nodes = len(atom_types)
        data.generation_formula = str(value).strip()
        return data
