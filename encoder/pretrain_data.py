"""Data loading, spectrum encoding, and batching for FragHub encoder pretraining."""
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import re
from collections import defaultdict
from typing import Optional
import random
from utils.collision_energy import energy_bin, collision_energy_bin

elements = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]  
precursor_mapping = {'[M+H]+': 0, '[M-H]-': 1}


def one_hot_encode_energy(val) -> torch.Tensor:
    """Bucket collision energy into 11 bins of 20 NCE each; clip ≥200 to the last bin."""
    total_dim = 11
    tensor = torch.zeros(total_dim)
    energy_class = energy_bin(val)
    tensor[energy_class] = 1
    return tensor


def one_hot_encode_precursor(val: str) -> torch.Tensor:
    """One-hot precursor type."""
    total_dim = len(precursor_mapping)
    tensor = torch.zeros(total_dim)
    tensor[precursor_mapping[val]] = 1
    return tensor


class positional_encoding:
    """Fixed sinusoidal positional encodings reused across items."""
    def __init__(self, size: int = 150):
        # 16-dim encoding per element-channel
        self.div_term = torch.exp(
            torch.arange(0, 16, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / 8)
        )
        self.encoding_template = torch.zeros(16)
        self.pos_dict = {}

        # Precompute encodings up to `size`
        for i in range(size):
            encoding = torch.zeros(16)
            encoding[0::2] = torch.sin(i * self.div_term)
            encoding[1::2] = torch.cos(i * self.div_term)
            self.pos_dict[i] = encoding

    def encode(self, n_atoms: int) -> torch.Tensor:
        """Return encoding for count `n_atoms` (negative maps to 0)."""
        if n_atoms < 0:
            return self.pos_dict[0]
        else:
            # NOTE: assumes n_atoms < `size`; if not, consider clamping or extending the table.
            return self.pos_dict[n_atoms]


def reorder_formula(formula: str) -> str:
    """Reorder formula as C, N, O, then others alphabetically; drop hydrogens."""
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    element_counts = defaultdict(int)
    for element, count in tokens:
        if element == 'H':
            continue  # Exclude hydrogen
        element_counts[element] += int(count) if count else 1

    priority = ['C', 'N', 'O']
    remaining = sorted(e for e in element_counts if e not in priority)
    ordered_elements = priority + remaining

    new_formula = ''
    for element in ordered_elements:
        if element in element_counts:
            count = element_counts[element]
            new_formula += f"{element}{count if count > 1 else ''}"
    return new_formula


class MSDataset(Dataset):
    """
    Returns a dict with:
      - 'batch_1': SOS features (precursor one-hot + energy one-hot) [B, 1, 2+11]
      - 'batch_2': padded per-peak encodings                       [B, max_peaks, 9*16]
      - 'batch_3': padding mask (True=masked)                      [B, max_peaks+1]
      - 'batch_4': list of SMILES                                  [B]
      - 'batch_5': information score list (float) or list of None          [B]
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        smiles_dict: dict = None,
        batch: bool = False,
        ordered: bool = False,
        ordered_sub_batch_size: int = 32,
        max_peaks: int = 128,
        device: str = "cuda"
    ):
        self.data = dataframe
        self.ordered = ordered
        self.batch = batch
        self.max_peaks = max_peaks
        self.device = device
        self.ordered_sub_batch_size = ordered_sub_batch_size

        if self.ordered:
            # Create blocks of similar formulas to help downstream batching
            dataframe['ReorderedFormula'] = dataframe['molecular_formula'].apply(reorder_formula)
            ordered_df = dataframe.sort_values(by=['ReorderedFormula'])

            unique_smiles = ordered_df['smiles'].drop_duplicates(keep='first').tolist()
            unique_smiles = [smi for smi in unique_smiles if smi in smiles_dict.keys()]
            subbatch_size = self.ordered_sub_batch_size
            self.ordered_smiles = [
                unique_smiles[i:i + subbatch_size]
                for i in range(0, len(unique_smiles), subbatch_size)
            ]

        self.smiles_dict = smiles_dict
        self.smiles = list(smiles_dict.keys())

        # Pre-allocations reused per item for speed
        self.padded_tensor_template = torch.zeros((max_peaks, 144), dtype=torch.float32)  # 9 elements * 16-d enc
        self.positional_encoding = positional_encoding()

    def encode_peaks(self, arr_list: np.ndarray) -> torch.Tensor:
        """Encode each peak element-count vector into concatenated sinusoidal embeddings."""
        total_dim = len(elements) * 16
        results = []
        for arr in arr_list:
            tensor = torch.zeros(total_dim)
            start_idx = 0
            for idx in range(len(elements)):
                value = arr[idx]
                tensor[start_idx:start_idx + 16] += self.positional_encoding.encode(value)
                start_idx += 16
            results.append(tensor)
        return torch.stack(results)

    def _convert_formula_array(self, array):
        """Pad/Mask encoded peaks to max length."""
        if isinstance(array, list):
            array = np.array(array)

        array = self.encode_peaks(array)
        n = array.shape[0]

        padded_tensor = self.padded_tensor_template.clone()  # avoid sharing across items
        padded_tensor[:n] = array

        # Mask: True means "ignore". First token is SOS → keep unmasked at position 0.
        mask = torch.ones(self.max_peaks + 1, dtype=torch.bool)
        mask[:n + 1] = 0
        return padded_tensor, mask

    def __len__(self):
        if self.ordered:
            return len(self.ordered_smiles)
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        Supports combinations of (batch, ordered):
        (True,  True)  : many spectra per SMILES, across an ordered sub-batch of SMILES
        (True,  False) : many spectra for a single SMILES
        (False, True)  : one spectrum per SMILES, across an ordered sub-batch of SMILES
        (False, False) : one spectrum for a single SMILES
        """

        def _process_row(sub_idx, collect_smiles=None):
            row = self.data.iloc[sub_idx]
            precursor_type = one_hot_encode_precursor(row['precursor_type'])
            collision_energy_nce = torch.zeros(11)
            collision_energy_nce[collision_energy_bin(row)] = 1
            spectrum = row['clean_spectrum_formula_array']
            
            if 'spectral_information_score' in row:
                information_score = torch.tensor([row['spectral_information_score']])
            else:
                information_score = torch.tensor([1])

            sos = torch.cat([precursor_type, collision_energy_nce], dim=0).view(1, -1)
            formula_array, mask = self._convert_formula_array(spectrum)

            return (
                sos.to(self.device),
                formula_array.to(self.device),
                mask.to(self.device),
                collect_smiles if collect_smiles is not None else None,
                information_score
            )

        # (True, True): iterate over an ordered list of SMILES; for each SMILES, collect *all* spectra
        if self.batch and self.ordered:
            smiles_list = self.ordered_smiles[idx]  # list[str]
            batch_sos, batch_formula_arrays, batch_masks, batch_smiles, batch_inf_scores = [], [], [], [], []

            for smi in smiles_list:
                for sub_idx in self.smiles_dict[smi]:
                    sos, fa, m, _, inf_scores = _process_row(sub_idx, collect_smiles=smi)
                    batch_sos.append(sos)
                    batch_formula_arrays.append(fa)
                    batch_masks.append(m)
                    batch_smiles.append(smi)  # repeat per spectrum
                    batch_inf_scores.append(inf_scores)

            return {
                'batch_1': torch.stack(batch_sos),
                'batch_2': torch.stack(batch_formula_arrays),
                'batch_3': torch.stack(batch_masks),
                'batch_4': batch_smiles,
                'batch_5': torch.stack(batch_inf_scores)
            }

        # (True, False): many spectra for a single SMILES
        if self.batch and not self.ordered:
            smi = self.smiles[idx]
            indices = self.smiles_dict[smi]

            if indices is None:
                return {
                'batch_1': torch.zeros(1,1),
                'batch_2': torch.zeros(1,1),
                'batch_3': torch.zeros(1,1),
                'batch_4': [smi],
                'batch_5': torch.zeros(1,1)
                }
                
            batch_sos, batch_formula_arrays, batch_masks, batch_smiles, batch_inf_scores = [], [], [], [], []
            for sub_idx in indices:
                sos, fa, m, _, inf_scores = _process_row(sub_idx, collect_smiles=smi)
                batch_sos.append(sos)
                batch_formula_arrays.append(fa)
                batch_masks.append(m)
                batch_smiles.append(smi)  # repeat per spectrum
                batch_inf_scores.append(inf_scores)

            return {
                'batch_1': torch.stack(batch_sos),
                'batch_2': torch.stack(batch_formula_arrays),
                'batch_3': torch.stack(batch_masks),
                'batch_4': batch_smiles,
                'batch_5': torch.stack(batch_inf_scores)
            }

        # (False, True): one spectrum per SMILES from an ordered sub-batch of SMILES
        if (not self.batch) and self.ordered:
            smiles_list = self.ordered_smiles[idx]  # list[str]
            # pick one spectrum index per SMILES
            indices = [random.choice(self.smiles_dict[smi]) for smi in smiles_list]

            batch_sos, batch_formula_arrays, batch_masks, batch_inf_scores = [], [], [], []
            for sub_idx in indices:
                sos, fa, m, _, inf_scores = _process_row(sub_idx)
                batch_sos.append(sos)
                batch_formula_arrays.append(fa)
                batch_masks.append(m)
                batch_inf_scores.append(inf_scores)

            return {
                'batch_1': torch.stack(batch_sos),
                'batch_2': torch.stack(batch_formula_arrays),
                'batch_3': torch.stack(batch_masks),
                'batch_4': smiles_list,          # one SMILES per selected spectrum
                'batch_5': torch.stack(batch_inf_scores)
            }

        # (False, False): one spectrum for a single SMILES
        smi = self.smiles[idx]
        sub_idx = random.choice(self.smiles_dict[smi])
        sos, fa, m, _, inf_scores = _process_row(sub_idx)

        return {
            'batch_1': sos.unsqueeze(0),           # shape: (1, 1, D)
            'batch_2': fa.unsqueeze(0),            # shape: (1, ...)
            'batch_3': m.unsqueeze(0),             # shape: (1, ...)
            'batch_4': [smi],                      # list of length 1
            'batch_5': inf_scores.unsqueeze(0)           # list of length 1 (float or None)
        }

def collate_fn(batch):
    sos, formula_arrays, masks, smiles, inf_scores = [], [], [], [], []

    for item in batch:
        sos.append(item['batch_1'])
        formula_arrays.append(item['batch_2'])
        masks.append(item['batch_3'])
        smiles.extend(item['batch_4'])
        inf_scores.append(item['batch_5'])
    
    return (
        torch.cat(sos),
        torch.cat(formula_arrays),
        torch.cat(masks),
        smiles,
        torch.cat(inf_scores)  # return None when not computing
    )


def load_data(
    data_path,
    smiles_path,
    split,
    split_path,
    batch_size: int = 16,
    shuffle_train: bool = True,
    batch: bool = False,  # if true, return all ms spectra of each molecule
    ordered: bool = False,  # if true, aranges batches according to similarity
    ordered_sub_batch_size = 32,  # if ordered is true, define sub batch size of similar molecules
    random_seed = 42,  # optional seed for random split and sampling
    device = "cuda",
):
    print (f"loading data from {data_path}, {smiles_path}, split={split}, batch_size={batch_size}, shuffle_train={shuffle_train}, batch={batch}, ordered={ordered}")
    # Load Parquet to pandas
    full_data = pq.read_table(data_path, use_threads=True).to_pandas()
    print("data was successfully loaded")

    # Load smiles -> graph, fp dictionary
    graph_dict = torch.load(smiles_path, weights_only=False)

    if split == 'random':
        smiles = list(graph_dict.keys())

        # Apply a reproducible shuffle if a seed is provided
        if random_seed is not None:
            rng = random.Random(random_seed)
            rng.shuffle(smiles)
        else:
            random.shuffle(smiles)

        N_smiles = len(smiles)
        train_smiles = smiles[:int(N_smiles * 0.9)]
        val_smiles = smiles[int(N_smiles * 0.9):int(N_smiles * 0.95)]
        test_smiles = smiles[int(N_smiles * 0.95):]
    
    elif split == 'predefined':
        import pickle
        
        with open(split_path, "rb") as f:
            split_dict = pickle.load(f)

        train_smiles, val_smiles, test_smiles = split_dict.values()
    
    # Build datasets; index list is at [0], remaining entries kept in graph_dict
    train_dataset = MSDataset(
        full_data,
        {i: graph_dict[i][0] for i in train_smiles},
        batch=batch,
        ordered=ordered,
        ordered_sub_batch_size=ordered_sub_batch_size,
        device=device,
    )
    val_dataset = MSDataset(
        full_data,
        {i: graph_dict[i][0] for i in val_smiles},
        batch=batch,
        ordered=ordered,
        ordered_sub_batch_size=ordered_sub_batch_size,
        device=device,
    )
    test_dataset = MSDataset(
        full_data,
        {i: graph_dict[i][0] for i in test_smiles},
        batch=batch,
        ordered=ordered,
        ordered_sub_batch_size=ordered_sub_batch_size,
        device=device,
    )
    graph_dict = {i: graph_dict[i][1:] for i in graph_dict}

    # Dataloaders (ordered mode reduces effective batch via 32-smiles subbatches)
    if ordered:
        effective_bs = max(1, int(batch_size / max(1, ordered_sub_batch_size)))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=effective_bs,
            shuffle=shuffle_train,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=effective_bs,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=effective_bs,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, graph_dict
