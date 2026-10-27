"""Rerank sampling caches or a complete feature table using external frozen models."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

from rerank.numpy_pickle_compat import install_numpy2_pickle_aliases
from utils.config import external_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="msg", choices=["msg", "nplib1"])
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--candidate-table", type=Path,
                        help="Full feature table with the frozen model's feature columns.")
    inputs.add_argument("--cache-glob", help="Quoted glob for current sampler candidates_rank*.jsonl.")
    parser.add_argument("--model-dir", type=Path,
                        help="Frozen model directory produced by rerank.train.")
    parser.add_argument("--data-root", type=Path, help="Contains labels.tsv, split.tsv and spec_files/.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-mode", choices=["direct"], default="direct")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and model feature contracts without scoring.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluate", action="store_true", help="Also compute UEP metrics; needs myopic-mces and CBC.")
    args = parser.parse_args()
    root = external_root() if not all([args.model_dir, args.data_root, args.output_dir]) else None
    model_dir = (args.model_dir or root / args.dataset / "output/rerank_training/models").resolve()
    out = (args.output_dir or root / args.dataset / "output/rerank").resolve()
    data = (args.data_root or root / args.dataset / "data").resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        parser.error("Keep generated output outside the source repository.")
    if args.workers < 1:
        parser.error("--workers must be positive")
    table = args.candidate_table.resolve() if args.candidate_table else out / "full_features.tsv"
    models = {key: model_dir / name for key, name in {
        "exact": "evidencerank_full_model.pkl",
        "stage_a": "stage_a_model.pkl",
        "ssr": "ssr_contrastive_tan_best.pt",
    }.items()}
    manifest_path = model_dir / 'manifest.json'
    required_paths = [*models.values(), manifest_path, model_dir / 'proxy_stats.json', data / 'labels.tsv', data / 'split.tsv']
    if args.candidate_table:
        required_paths.append(table)
    for path in required_paths:
        if not path.is_file():
            parser.error(f"Missing external input: {path}")
    if not (data / "spec_files").is_dir():
        parser.error(f"Missing spectra directory: {data / 'spec_files'}")
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('dataset') != args.dataset or manifest.get('feature_mode') != 'direct'
            or manifest.get('feature_schema') != 'fragment_v2_direct_241_v1'):
        parser.error('Model bundle dataset/feature recipe does not match this run')
    from rerank.training_common import sha256
    expected_files = {p.name for p in models.values()} | {'proxy_stats.json'}
    if not expected_files <= set(manifest.get('sha256', {})):
        parser.error('Incomplete frozen model manifest')
    for name in expected_files:
        if sha256(model_dir / name) != manifest['sha256'][name]:
            parser.error(f'Model bundle checksum mismatch: {name}')
    base_weight = float(manifest['ssr_base_weight'])
    if not 0 <= base_weight <= 1:
        parser.error('Invalid frozen SSR fusion weight')
    install_numpy2_pickle_aliases()
    import numpy as np
    import pandas as pd
    from rerank import score
    from rerank.contracts import validate_metadata, validate_model_features, validate_feature_table

    def execute(module, argv):
        subprocess.run([sys.executable, "-m", module, *map(str, argv)], cwd=repo, check=True)

    with models["exact"].open("rb") as handle:
        exact = pickle.load(handle)
    with models["stage_a"].open("rb") as handle:
        stage_a = pickle.load(handle)
    if abs(float(stage_a['selected_exact_weight']) - float(manifest['stage_a_exact_weight'])) > 1e-12:
        parser.error('Stage A weight disagrees with frozen manifest')
    validate_model_features(exact["features"])
    validate_model_features(stage_a["features"])
    if args.cache_glob:
        import glob
        paths = sorted(glob.glob(args.cache_glob))
        if not paths:
            parser.error(f"No caches match: {args.cache_glob}")
        ids = []
        active_ids = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if item.get("type") in ("mol_candidates", "input_failure"):
                            ids.append(str(item["spec_id"]))
                            if item.get("pred", {}).get("connected_smiles_nonisomeric_counts"):
                                active_ids.append(str(item["spec_id"]))
        validate_metadata(ids, data, required_spectra=active_ids)
    else:
        frame_ids = pd.read_csv(table, sep="\t", usecols=["spec_id"], dtype=str)
        validate_metadata(frame_ids.spec_id.drop_duplicates().tolist(), data)
        validate_feature_table(table, [*exact["features"], *stage_a["features"]])
    if args.check_only:
        print("Inputs and ranking model feature contracts checked. No ranking was performed.")
        return
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty; choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    if args.cache_glob:
        from rerank.build_candidates import build_candidates
        summary = build_candidates(args.cache_glob, out / "candidates.tsv",
                                   out / "spectrum_status.tsv", out / "candidate_summary.json")
        (out / "feature_recipe.json").write_text(json.dumps({
            "mode": args.feature_mode,
            "description": "Deterministic fragment-v2 rules with frozen training statistics.",
            "manifest_sha256": sha256(manifest_path),
            "model_sha256": {k: score.sha256_file(p) for k, p in models.items()},
        }, indent=2), encoding="utf-8")
        if summary["n_candidates"]:
            execute("rerank.features", [
                "--candidate-structures", out / "candidates.tsv", "--msg-root", data,
                "--out-table", out / "evidence_features.tsv",
                "--summary-json", out / "evidence_summary.json", "--accept-all-valid-rows",
                "--top-peaks", 30, "--ppm", 20, "--abs-tol", 0.01,
                "--max-single-cuts", 48, "--max-double-cut-pairs", 12,
                "--max-ring-open-pairs", 8, "--max-ion-hypotheses", 256,
            ])
            execute("rerank.assemble_features", [
                "--base-table", out / "candidates.tsv", "--erl-glob", out / "evidence_features.tsv",
                "--out-table", table,
                "--summary-json", out / "feature_summary.json", "--workers", args.workers,
                "--data-root", data, "--model-dir", model_dir,
            ])
        else:
            pd.DataFrame(columns=["spec_id", "candidate_hash", "score"]).to_csv(
                out / "scores.tsv", sep="\t", index=False)
            pd.DataFrame(columns=["spec_id", "rank", "candidate_canonical_smiles_nostereo",
                                  "candidate_hash", "score"]).to_csv(
                out / "ranked_candidates.tsv", sep="\t", index=False)
            print("No valid candidates. See spectrum_status.tsv; no ranking model was applied.")
            if args.evaluate:
                execute("rerank.evaluate", ["--candidate-table", out / "candidates.tsv",
                        "--scores", out / "scores.tsv", "--split", data / "split.tsv",
                        "--labels", data / "labels.tsv", "--output-dir", out / "metrics",
                        "--workers", args.workers, "--dataset", args.dataset])
            return
        validate_feature_table(table, [*exact["features"], *stage_a["features"]])
    features = exact["features"]
    required = list(dict.fromkeys(["spec_id", "candidate_hash", *features]))
    frame = pd.read_csv(table, sep="\t", usecols=required,
                        dtype={"spec_id": str, "candidate_hash": str}, low_memory=False)
    if frame.duplicated(["spec_id", "candidate_hash"]).any():
        raise ValueError("Duplicate spectrum/candidate keys")
    matrix = np.column_stack([pd.to_numeric(frame[f], errors="coerce").fillna(0).to_numpy(np.float32)
                              for f in features])
    exact_scores = out / "evidencerank_scores.tsv"
    frame[["spec_id", "candidate_hash"]].assign(
        score=exact["model"].predict(matrix)).to_csv(exact_scores, sep="\t", index=False)
    del frame, matrix, exact

    def apply(argv):
        parsed = score.build_parser().parse_args([str(v) for v in argv])
        parsed.func(parsed)

    stage_scores = out / "stage_a_scores.tsv"
    apply(["stage-a", "--candidate-table", table, "--model-pkl", models["stage_a"],
           "--evidencerank-scores", exact_scores, "--out-scores", stage_scores,
           "--declared-split", "test", "--exact-weight", stage_a["selected_exact_weight"]])
    cache = out / "ssr_cache"
    apply(["prepare-test", "--candidate-table", table, "--labels", data / "labels.tsv",
           "--split-table", data / "split.tsv", "--spec-dir", data / "spec_files",
           "--out-dir", cache, "--declared-split", "test", "--workers", args.workers,
           "--max-peaks", str(manifest["max_peaks"])])
    digest = hashlib.sha256()
    with models["ssr"].open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    neural_scores = out / "ssr_scores.tsv"
    apply(["score-ssr", "--candidate-cache", cache, "--checkpoint", models["ssr"],
           "--expected-checkpoint-sha256", digest.hexdigest(), "--out-scores", neural_scores,
           "--device", args.device])
    fused = out / "scores.tsv"
    apply(["fuse", "--candidate-cache", cache, "--stage-a-scores", stage_scores,
           "--neural-scores", neural_scores, "--out-scores", fused,
           "--declared-split", "test", "--base-weight", base_weight])
    candidates = pd.read_csv(table, sep="\t", usecols=[
        "spec_id", "candidate_hash", "candidate_canonical_smiles_nostereo", "candidate_rank_freq"],
        dtype={"spec_id": str, "candidate_hash": str})
    ranked = candidates.merge(pd.read_csv(fused, sep="\t", dtype={"spec_id": str, "candidate_hash": str}),
                              on=["spec_id", "candidate_hash"], validate="one_to_one")
    ranked = ranked.sort_values(["spec_id", "score", "candidate_rank_freq", "candidate_hash"],
                                ascending=[True, False, True, True], kind="mergesort")
    ranked.insert(1, "rank", ranked.groupby("spec_id", sort=False).cumcount() + 1)
    ranked.to_csv(out / "ranked_candidates.tsv", sep="\t", index=False)
    if args.evaluate:
        subprocess.run([sys.executable, "-m", "rerank.evaluate", "--candidate-table", str(table),
                        "--scores", str(fused), "--split", str(data / "split.tsv"),
                        "--labels", str(data / "labels.tsv"), "--output-dir", str(out / "metrics"),
                        "--workers", str(args.workers), "--dataset", args.dataset], check=True,
                       cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
