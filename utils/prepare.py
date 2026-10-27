"""Convert external embedding caches and supervised rows to the EQ packed format."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from utils.formula import parse_formula, acquisition_indices
from utils.graph import graph_from_smiles
from utils.sample import sha256
from utils.artifacts import resolve


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to mix new cache with existing files: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def embeddings(args):
    values = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(values, dict) or not values:
        raise ValueError("Expected a nonempty key -> 512-vector embedding dictionary")
    if not all(isinstance(k, str) for k in values):
        raise ValueError("Embedding keys must be strings")
    root = fresh_dir(args.output)
    keys = sorted(values)
    array = np.lib.format.open_memmap(root / "features.npy", mode="w+", dtype=np.float32,
                                     shape=(len(keys), 512))
    for i, key in enumerate(keys):
        v = torch.as_tensor(values[key], dtype=torch.float32).reshape(-1)
        if v.shape != (512,) or not torch.isfinite(v).all() or v.norm() == 0:
            raise ValueError(f"Invalid embedding: {key}")
        array[i] = v.numpy()
    array.flush()
    (root / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    manifest = {"tower": args.tower, "dim": 512, "dtype": "float32", "normalization": "NONE",
                "count": len(keys), "encoder_sha256": sha256(resolve(args.encoder_checkpoint, 'encoder'))}
    (root / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_embedding(root, tower):
    root = Path(root)
    manifest = json.loads((root / "CACHE_MANIFEST.json").read_text())
    if manifest.get("tower") != tower or manifest.get("normalization") != "NONE":
        raise ValueError(f"Wrong embedding tower/normalization: {root}")
    keys = json.loads((root / "keys.json").read_text())
    if not isinstance(keys, list) or not all(isinstance(x, str) for x in keys) or len(keys) != len(set(keys)):
        raise ValueError("This converter requires string-list keys (Zmol: canonical SMILES)")
    array = np.load(root / "features.npy", mmap_mode="r")
    if array.shape != (len(keys), 512) or array.dtype != np.float32:
        raise ValueError("Expected an unnormalized FP32 512-dimensional cache")
    return manifest, {k: i for i, k in enumerate(keys)}, array


def pack(args):
    """Input rows are TRAIN/VAL only; generation has a separate command."""
    delimiter = "," if Path(args.rows).suffix == ".csv" else "\t"
    with open(args.rows, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=delimiter))
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("No rows in requested split")
    zm, zk, zv = load_embedding(args.zmol_root, "zmol")
    sm, sk, sv = (load_embedding(args.zms_root, "zms") if args.stage == "ms2mol"
                  else (None, None, None))
    if sm and sm.get("encoder_sha256") != zm.get("encoder_sha256"):
        raise ValueError("Zms and Zmol came from different encoders")
    root = fresh_dir(args.output)
    db = sqlite3.connect(root / "records.sqlite")
    db.execute("CREATE TABLE records (idx INTEGER PRIMARY KEY, sample_id TEXT UNIQUE, formula TEXT,"
               "n INTEGER,nodes BLOB,bonds BLOB,zms_index INTEGER,zmol_index INTEGER,"
               "instrument INTEGER,adduct INTEGER)")
    counts = np.empty(len(rows), dtype=np.int32)
    features = (np.lib.format.open_memmap(root / "features.npy", mode="w+", dtype=np.float32,
                                        shape=(len(rows), 512)) if args.stage == "graph2mol" else None)
    for idx, row in enumerate(rows):
        sid = str(row.get("sample_id") or row.get("spec") or row.get("mol_id") or "")
        if not sid:
            raise ValueError("Each row needs sample_id (or spec/mol_id)")
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid supervised SMILES: {sid}")
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        formula = row.get("formula") or rdMolDescriptors.CalcMolFormula(mol)
        if parse_formula(formula, adapter="graph2mol").counts != parse_formula(
                rdMolDescriptors.CalcMolFormula(mol), adapter="graph2mol").counts:
            raise ValueError(f"Formula and supervised structure disagree: {sid}")
        gi = zk[smiles]
        si = sk[sid] if sm else -1
        if args.stage == "ms2mol":
            inst, ion = acquisition_indices(row.get("instrument"), row.get("adduct") or row.get("ionization"),
                                            args.adapter)
        else:
            inst, ion = 2, 0
        z = torch.tensor(np.array(zv[gi]))
        graph = graph_from_smiles(sid, smiles, torch.zeros(512) if si < 0 else torch.tensor(np.array(sv[si])),
                                  z, split=args.split, instrument_idx=inst, ionization_idx=ion)
        n = len(graph.node_type)
        selected = graph.halfedge_type > 0
        bonds = torch.cat([graph.halfedge_index[:, selected].t(), graph.halfedge_type[selected, None]], dim=1)
        db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (idx, sid, formula, n, graph.node_type.numpy().astype(np.uint8).tobytes(),
                    bonds.numpy().astype(np.uint8).tobytes(), si, gi, inst, ion))
        counts[idx] = n
        if features is not None:
            features[idx] = zv[gi]
    db.commit()
    db.close()
    np.save(root / "node_counts.npy", counts)
    if features is not None:
        features.flush()
    manifest = {"qualified_rows": len(rows), "role": args.split, "encoder_sha256": zm["encoder_sha256"],
                "dim": 512, "dtype": "float32", "normalization": "NONE"}
    if sm:
        manifest.update(zms_root=str(Path(args.zms_root).resolve()), zmol_root=str(Path(args.zmol_root).resolve()),
                        zms_manifest_sha256=sha256(Path(args.zms_root)/'CACHE_MANIFEST.json'),
                        zmol_manifest_sha256=sha256(Path(args.zmol_root)/'CACHE_MANIFEST.json'))
    (root / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generation_inputs(args):
    """Project metadata only; never derive a generation formula from SMILES."""
    with open(args.split_table, newline="", encoding="utf-8") as f:
        ids = {r["name"] for r in csv.DictReader(f, delimiter="\t") if r["split"] == args.split}
    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "formula", "formula_kind", "formula_source", "instrument", "adduct", "split"]
    seen = set()
    with output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for r in rows:
            sid = r.get("spec") or r.get("identifier") or r.get("sample_id")
            if sid not in ids:
                continue
            if sid in seen:
                raise ValueError(f"Duplicate label row: {sid}")
            seen.add(sid)
            writer.writerow(dict(sample_id=sid, formula=r.get("formula", ""),
                                 formula_kind="neutral", formula_source="provided_metadata",
                                 instrument=r.get("instrument", "NONE"), adduct=r.get("ionization", ""),
                                 split=args.split))
    if seen != ids:
        raise ValueError(f"Missing {len(ids - seen)} split metadata rows")


def pipeline(args):
    """Pack prepared encoder caches for both decoder stages and test generation."""
    import tempfile
    from types import SimpleNamespace
    from encoder.config import load, path
    from utils.config import external_root
    from utils.artifacts import sha256 as digest
    cfg = load(args.encoder_config, dataset=args.dataset)
    dataset = cfg['dataset']; selected = cfg['datasets'][dataset]
    root = external_root() / dataset
    enc = digest(resolve(root / 'embeddings/encoder.json', 'encoder'))
    for tower in ('zms', 'zmol'):
        manifest, _, _ = load_embedding(root/'embeddings'/tower, tower)
        if manifest.get('encoder_sha256') != enc:
            raise ValueError(f'{tower} does not match the selected encoder')
    if (root/'packed').exists() or (root/'generation_inputs.tsv').exists():
        raise FileExistsError('Prepared decoder outputs already exist; choose a new asset directory or explicitly manage old outputs')
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        dest = Path(tmp)
        for stage in ('graph2mol', 'ms2mol'):
            rows = (args.graph_rows or path(selected['graph_rows'])) if stage == 'graph2mol' else path(selected['prepared_dir'])/'manifest.tsv'
            for split in ('train', 'val'):
                pack(SimpleNamespace(rows=rows, split=split, stage=stage, adapter=dataset,
                     zms_root=root/'embeddings/zms', zmol_root=root/'embeddings/zmol',
                     output=dest/'packed'/stage/split))
        data = path(selected['data_root'])
        generation_inputs(SimpleNamespace(labels=data/'labels.tsv', split_table=data/'split.tsv',
                          split='test', output=dest/'generation_inputs.tsv'))
        (dest/'packed').rename(root/'packed')
        (dest/'generation_inputs.tsv').rename(root/'generation_inputs.tsv')
    print(f'Decoder inputs ready: {root}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("embeddings")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--tower", choices=["zms", "zmol"], required=True)
    q.add_argument("--encoder-checkpoint", required=True)
    q.set_defaults(func=embeddings)
    q = sub.add_parser("pack")
    q.add_argument("--rows", required=True, help="CSV/TSV with split, sample_id/spec, smiles, formula and acquisition")
    q.add_argument("--output", required=True)
    q.add_argument("--stage", choices=["graph2mol", "ms2mol"], required=True)
    q.add_argument("--split", choices=["train", "val"], required=True)
    q.add_argument("--adapter", choices=["msg", "nplib1"], default="msg")
    q.add_argument("--zmol-root", required=True)
    q.add_argument("--zms-root")
    q.set_defaults(func=pack)
    q = sub.add_parser("generation-inputs")
    q.add_argument("--labels", required=True)
    q.add_argument("--split-table", required=True)
    q.add_argument("--split", choices=["train", "val", "test"], default="test")
    q.add_argument("--output", required=True)
    q.set_defaults(func=generation_inputs)
    q = sub.add_parser('pipeline', help='Prepare both decoder stages and generation inputs from exported caches')
    q.add_argument('--encoder-config', default='configs/encoder.yml')
    q.add_argument('--dataset', choices=['msg', 'nplib1'])
    q.add_argument('--graph-rows', type=Path)
    q.set_defaults(func=pipeline)
    args = p.parse_args()
    if args.command == "pack" and args.stage == "ms2mol" and not args.zms_root:
        p.error("MS2Mol packing requires --zms-root")
    args.func(args)


if __name__ == "__main__":
    main()
