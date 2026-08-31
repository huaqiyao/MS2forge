

from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType

RDLogger.DisableLog('rdApp.*')



DEFAULT_ATOMIC_NUMBERS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]
_BT_MAP = {0: None, 1: BondType.SINGLE, 2: BondType.DOUBLE,
           3: BondType.TRIPLE, 4: BondType.AROMATIC}


def edges_to_mol(node_types, halfedge_index, edge_types, atomic_numbers=None):


    if atomic_numbers is None:
        atomic_numbers = DEFAULT_ATOMIC_NUMBERS
    mol = Chem.RWMol()
    nt = node_types.tolist() if hasattr(node_types, 'tolist') else list(node_types)
    for n in nt:
        if n < 0 or n >= len(atomic_numbers):
            return None
        mol.AddAtom(Chem.Atom(atomic_numbers[n]))
    he_src = halfedge_index[0].tolist() if hasattr(halfedge_index[0], 'tolist') else list(halfedge_index[0])
    he_dst = halfedge_index[1].tolist() if hasattr(halfedge_index[1], 'tolist') else list(halfedge_index[1])
    et = edge_types.tolist() if hasattr(edge_types, 'tolist') else list(edge_types)
    for k, e in enumerate(et):
        if e == 0:
            continue
        bt = _BT_MAP.get(int(e))
        if bt is None:
            continue
        try:
            mol.AddBond(int(he_src[k]), int(he_dst[k]), bt)
        except Exception:
            return None
    try:
        out = mol.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def edges_to_canonical_smiles(node_types, halfedge_index, edge_types,
                               atomic_numbers=None):


    mol = edges_to_mol(node_types, halfedge_index, edge_types, atomic_numbers)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_to_inchikey(mol):

    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def mol_to_inchi(mol):
    """RDKit Mol -> standard InChI; invalid or unsupported molecules return None."""
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
    except Exception:
        return None
    return inchi or None


def is_diffms_valid_mol(mol):
    """DiffMS validity filter: sanitizable and single-fragment molecule."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(
            mol, asMols=True, sanitizeFrags=True
        )
    except Exception:
        return False
    return len(mol_frags) <= 1


def edges_to_inchikey(node_types, halfedge_index, edge_types, atomic_numbers=None):
    """node_types + halfedge_index + edge_types  ->  InChIKey."""
    return mol_to_inchikey(edges_to_mol(node_types, halfedge_index, edge_types, atomic_numbers))


def edges_to_diffms_inchi(node_types, halfedge_index, edge_types, atomic_numbers=None):
    """node_types + halfedge_index + edge_types -> DiffMS-filtered InChI."""
    mol = edges_to_mol(node_types, halfedge_index, edge_types, atomic_numbers)
    if not is_diffms_valid_mol(mol):
        return None
    return mol_to_inchi(mol)


def mol_to_diffms_2d(mol):
    """Return the stage-II identity used for the DiffMS-aligned 2D metric.

    Candidates must sanitize and contain exactly one connected fragment.  The
    identity intentionally ignores stereochemistry because FLASH predicts only
    the non-stereochemical 2D graph.
    """
    if not is_diffms_valid_mol(mol):
        return None
    try:
        return Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        )
    except Exception:
        return None


def edges_to_diffms_2d(node_types, halfedge_index, edge_types, atomic_numbers=None):
    """Edges -> valid, connected, canonical non-isomeric SMILES."""
    return mol_to_diffms_2d(
        edges_to_mol(node_types, halfedge_index, edge_types, atomic_numbers)
    )


def smiles_to_mol(smiles):
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def counter_pairs(values):
    return [[item, int(count)] for item, count in Counter(values).most_common()]


def topk_hit_strict(pred_seqs, true_seq, topk_list, **kwargs):

    pred_tuples = [tuple(p.tolist()) for p in pred_seqs]
    true_tuple = tuple(true_seq.tolist())
    counter = Counter(pred_tuples)
    sorted_unique = [t for t, _ in counter.most_common()]
    return {k: (true_tuple in sorted_unique[:k]) for k in topk_list}


def topk_hit_isomorphic(pred_seqs, true_seq, topk_list,
                         node_types=None, halfedge_index=None,
                         atomic_numbers=None, true_smiles=None):



    if node_types is None or halfedge_index is None:
        raise ValueError("isomorphic mode requires  node_types / halfedge_index")
    if true_smiles is not None:
        try:
            _m = Chem.MolFromSmiles(true_smiles)
            true_smi = Chem.MolToSmiles(_m, canonical=True) if _m else None
        except Exception:
            true_smi = None
    else:
        true_smi = edges_to_canonical_smiles(node_types, halfedge_index, true_seq, atomic_numbers)
    pred_smis = []
    for p in pred_seqs:
        smi = edges_to_canonical_smiles(node_types, halfedge_index, p, atomic_numbers)
        if smi is not None:
            pred_smis.append(smi)
    counter = Counter(pred_smis)
    sorted_unique = [s for s, _ in counter.most_common()]
    if true_smi is None:
        return topk_hit_strict(pred_seqs, true_seq, topk_list)
    return {k: (true_smi in sorted_unique[:k]) for k in topk_list}


def topk_hit_inchikey(pred_seqs, true_seq, topk_list,
                       node_types=None, halfedge_index=None,
                       atomic_numbers=None, true_smiles=None):

    if node_types is None or halfedge_index is None:
        raise ValueError("inchikey mode requires  node_types / halfedge_index")
    if true_smiles is not None:
        try:
            true_key = mol_to_inchikey(Chem.MolFromSmiles(true_smiles))
        except Exception:
            true_key = None
    else:
        true_key = edges_to_inchikey(node_types, halfedge_index, true_seq, atomic_numbers)

    pred_keys = []
    for p in pred_seqs:
        key = edges_to_inchikey(node_types, halfedge_index, p, atomic_numbers)
        if key:
            pred_keys.append(key)
    counter = Counter(pred_keys)
    sorted_unique = [s for s, _ in counter.most_common()]
    if true_key is None:
        return topk_hit_strict(pred_seqs, true_seq, topk_list)
    return {k: (true_key in sorted_unique[:k]) for k in topk_list}


def topk_hit_diffms_inchi(pred_seqs, true_seq, topk_list,
                          node_types=None, halfedge_index=None,
                          atomic_numbers=None, true_smiles=None):
    """DiffMS exact-match protocol: valid connected candidates -> InChI -> frequency rank."""
    if node_types is None or halfedge_index is None:
        raise ValueError("diffms_inchi mode requires  node_types / halfedge_index")
    if true_smiles is not None:
        true_inchi = mol_to_inchi(smiles_to_mol(true_smiles))
    else:
        true_inchi = mol_to_inchi(
            edges_to_mol(node_types, halfedge_index, true_seq, atomic_numbers)
        )

    pred_inchis = []
    for p in pred_seqs:
        inchi = edges_to_diffms_inchi(
            node_types, halfedge_index, p, atomic_numbers
        )
        if inchi:
            pred_inchis.append(inchi)
    sorted_unique = [inchi for inchi, _ in Counter(pred_inchis).most_common()]
    if true_inchi is None:
        return topk_hit_strict(pred_seqs, true_seq, topk_list)
    return {k: (true_inchi in sorted_unique[:k]) for k in topk_list}


def topk_hit_diffms_2d(pred_seqs, true_seq, topk_list,
                       node_types=None, halfedge_index=None,
                       atomic_numbers=None, true_smiles=None):
    """Stage-II metric: valid connected non-isomeric 2D candidates.

    Invalid/disconnected generations are removed, equivalent candidates are
    merged by canonical non-isomeric SMILES, and unique candidates are ranked
    by sample frequency.  ``Counter.most_common`` preserves first-generation
    order for equal frequencies, matching the established DiffMS-style cache
    re-evaluation.
    """
    if node_types is None or halfedge_index is None:
        raise ValueError("diffms_2d mode requires  node_types / halfedge_index")
    if true_smiles is not None:
        true_repr = mol_to_diffms_2d(smiles_to_mol(true_smiles))
    else:
        true_repr = edges_to_diffms_2d(
            node_types, halfedge_index, true_seq, atomic_numbers
        )

    pred_reprs = []
    for pred in pred_seqs:
        pred_repr = edges_to_diffms_2d(
            node_types, halfedge_index, pred, atomic_numbers
        )
        if pred_repr is not None:
            pred_reprs.append(pred_repr)
    sorted_unique = [value for value, _ in Counter(pred_reprs).most_common()]
    if true_repr is None:
        return topk_hit_strict(pred_seqs, true_seq, topk_list)
    return {k: (true_repr in sorted_unique[:k]) for k in topk_list}


def build_candidate_cache_record(pred_seqs, true_seq,
                                 node_types=None, halfedge_index=None,
                                 atomic_numbers=None, true_smiles=None):
    """Summarize generated candidates so metrics can be recomputed without sampling."""
    if node_types is None or halfedge_index is None:
        raise ValueError("candidate cache requires  node_types / halfedge_index")

    true_mol = smiles_to_mol(true_smiles)
    if true_mol is None:
        true_mol = edges_to_mol(node_types, halfedge_index, true_seq, atomic_numbers)
    true_smiles_iso = None
    true_smiles_noniso = None
    if true_mol is not None:
        try:
            true_smiles_iso = Chem.MolToSmiles(true_mol, canonical=True, isomericSmiles=True)
            true_smiles_noniso = Chem.MolToSmiles(true_mol, canonical=True, isomericSmiles=False)
        except Exception:
            pass

    valid_inchis = []
    valid_inchikeys = []
    valid_smiles = []
    connected_inchis = []
    connected_inchikeys = []
    connected_smiles = []
    connected_smiles_noniso = []
    invalid = 0
    disconnected = 0
    valid_count = 0
    connected_count = 0

    for p in pred_seqs:
        mol = edges_to_mol(node_types, halfedge_index, p, atomic_numbers)
        if mol is None:
            invalid += 1
            continue
        valid_count += 1
        key = mol_to_inchikey(mol)
        if key:
            valid_inchikeys.append(key)
        try:
            valid_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        except Exception:
            pass
        inchi = mol_to_inchi(mol)
        if inchi:
            valid_inchis.append(inchi)

        if is_diffms_valid_mol(mol):
            connected_count += 1
            if inchi:
                connected_inchis.append(inchi)
            if key:
                connected_inchikeys.append(key)
            try:
                connected_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
                connected_smiles_noniso.append(
                    Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                )
            except Exception:
                pass
        else:
            disconnected += 1

    return {
        'schema': 'flash_candidate_cache.v1',
        'type': 'mol_candidates',
        'protocol': 'diffms_2d',
        'true': {
            'input_smiles': true_smiles,
            'inchi': mol_to_inchi(true_mol),
            'inchikey': mol_to_inchikey(true_mol),
            'canonical_smiles': true_smiles_iso,
            'canonical_smiles_nonisomeric': true_smiles_noniso,
        },
        'pred': {
            'n_generated': len(pred_seqs),
            'n_valid': valid_count,
            'n_valid_connected': connected_count,
            'n_invalid': invalid,
            'n_disconnected': disconnected,
            'diffms_inchi_counts': counter_pairs(connected_inchis),
            'connected_inchikey_counts': counter_pairs(connected_inchikeys),
            'connected_smiles_counts': counter_pairs(connected_smiles),
            'connected_smiles_nonisomeric_counts': counter_pairs(connected_smiles_noniso),
            'valid_inchi_counts': counter_pairs(valid_inchis),
            'valid_inchikey_counts': counter_pairs(valid_inchikeys),
            'valid_smiles_counts': counter_pairs(valid_smiles),
        },
    }


def topk_hit_for_mol(pred_seqs, true_seq, topk_list, mode='strict',
                     node_types=None, halfedge_index=None, atomic_numbers=None,
                     true_smiles=None):




    if mode == 'strict':
        return topk_hit_strict(pred_seqs, true_seq, topk_list)
    elif mode == 'isomorphic':
        return topk_hit_isomorphic(pred_seqs, true_seq, topk_list,
                                    node_types=node_types,
                                    halfedge_index=halfedge_index,
                                    atomic_numbers=atomic_numbers,
                                    true_smiles=true_smiles)
    elif mode == 'inchikey':
        return topk_hit_inchikey(pred_seqs, true_seq, topk_list,
                                  node_types=node_types,
                                  halfedge_index=halfedge_index,
                                  atomic_numbers=atomic_numbers,
                                  true_smiles=true_smiles)
    elif mode == 'diffms_inchi':
        return topk_hit_diffms_inchi(pred_seqs, true_seq, topk_list,
                                     node_types=node_types,
                                     halfedge_index=halfedge_index,
                                     atomic_numbers=atomic_numbers,
                                     true_smiles=true_smiles)
    elif mode == 'diffms_2d':
        return topk_hit_diffms_2d(pred_seqs, true_seq, topk_list,
                                  node_types=node_types,
                                  halfedge_index=halfedge_index,
                                  atomic_numbers=atomic_numbers,
                                  true_smiles=true_smiles)
    else:
        raise ValueError(
            "eval_mode must be one of  strict / isomorphic / inchikey / "
            f"diffms_inchi / diffms_2d, ; received  {mode!r}"
        )
