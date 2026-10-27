"""Rerank candidates from the output directory selected in configs/rerank.yml."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from utils.config import external_root, load_yaml

REPO = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=["msg", "nplib1"],
                        help="Override the dataset selected in the rerank YAML.")
    parser.add_argument("--config", type=Path, default=REPO / "configs/rerank.yml",
                        help="Rerank dataset, paths and model bundle configuration.")
    parser.add_argument("--assets", help="Override MS2FORGE_ASSETS.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--cache-glob")
    inputs.add_argument("--candidate-table", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-mode", choices=["direct"], default="direct")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.assets:
        os.environ["MS2FORGE_ASSETS"] = args.assets
    try:
        with args.config.open(encoding="utf-8") as handle:
            cfg = load_yaml(handle, dataset=args.dataset)
        root = external_root()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    dataset = cfg["dataset"]
    cmd = [sys.executable, "-m", "rerank.run", dataset]
    if args.candidate_table:
        cmd += ["--candidate-table", str(args.candidate_table.resolve())]
    else:
        cache = args.cache_glob or cfg["candidate_globs"]["test"]
        # Resolve user-supplied relative paths before switching the child cwd.
        cmd += ["--cache-glob", str(Path(cache).resolve()),
                "--feature-mode", args.feature_mode]
    for option, path in (
        ("--model-dir", args.model_dir or Path(cfg["output_dir"]) / "models"),
        ("--data-root", args.data_root or Path(cfg["data_root"])),
        ("--output-dir", args.output_dir or root / dataset / "output/rerank"),
    ):
        cmd += [option, str(path.resolve())]
    cmd += ["--device", args.device or cfg["device"], "--workers", str(args.workers or cfg["workers"])]
    for option in ("evaluate", "check_only"):
        if getattr(args, option):
            cmd.append("--" + option.replace("_", "-"))
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
