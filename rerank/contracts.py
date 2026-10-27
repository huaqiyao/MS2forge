"""Validate ranking inputs without consuming reference structures or target labels."""
from pathlib import Path

import numpy as np
import pandas as pd


def validate_model_features(features):
    if not features or len(features) != len(set(features)):
        raise ValueError("Frozen model has an empty or duplicate feature list")
    forbidden = ("true", "truth", "target", "tanimoto_to_", "mces_to_", "is_correct", "hit@")
    bad = [name for name in features if any(token in str(name).lower() for token in forbidden)]
    if bad:
        raise ValueError(f"Reference-answer fields cannot be ranking inputs: {bad}")


def validate_feature_table(path, features):
    columns = list(dict.fromkeys(features))
    validate_model_features(columns)
    required = ["spec_id", "candidate_hash", "candidate_rank_freq",
                "candidate_canonical_smiles", "candidate_canonical_smiles_nostereo"]
    frame = pd.read_csv(path, sep="\t", usecols=list(dict.fromkeys(required + columns)),
                        dtype={"spec_id": str, "candidate_hash": str}, keep_default_na=False)
    if frame.empty:
        raise ValueError("The supplied full feature table is empty")
    if frame.duplicated(["spec_id", "candidate_hash"]).any():
        raise ValueError("Duplicate candidate keys")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing/non-finite frozen feature: {column}; do not replace it with zero")


def validate_metadata(ids, data_root, required_spectra=None):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate spectrum IDs in input")
    data_root = Path(data_root)
    labels = pd.read_csv(data_root / "labels.tsv", sep="\t", dtype=str, keep_default_na=False,
                         usecols=["spec", "formula", "ionization", "instrument"])
    split = pd.read_csv(data_root / "split.tsv", sep="\t", dtype=str, keep_default_na=False,
                        usecols=["name", "split"])
    if labels.spec.duplicated().any() or split.name.duplicated().any():
        raise ValueError("Duplicate metadata/split IDs")
    metadata = labels.set_index("spec")
    roles = split.set_index("name")["split"]
    required_spectra = set(ids) if required_spectra is None else set(required_spectra)
    for sid in ids:
        if sid not in roles.index or roles[sid] != "test":
            raise ValueError(f"{sid}: not in the declared test split")
        if sid not in required_spectra:
            continue
        if sid not in metadata.index:
            raise ValueError(f"{sid}: missing spectrum metadata")
        if not metadata.loc[sid, "formula"] or not metadata.loc[sid, "ionization"]:
            raise ValueError(f"{sid}: formula and ionization metadata are required")
        if Path(sid).name != sid or "/" in sid or "\\" in sid:
            raise ValueError(f"Invalid spectrum ID: {sid}")
        if not (data_root / "spec_files" / f"{sid}.ms").is_file():
            raise FileNotFoundError(f"Missing spectrum: {sid}")
