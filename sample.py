"""Generate molecules using configs/sample.yml (8 GPUs by default)."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=["msg", "nplib1"],
                        help="Override the dataset selected in the YAML.")
    parser.add_argument("--config", type=Path, default=REPO / "configs/sample.yml")
    parser.add_argument("--assets", help="Override MS2FORGE_ASSETS.")
    # Use --gpus 1 for a single GPU.
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("--gpus must be at least 1")
    if args.assets:
        os.environ["MS2FORGE_ASSETS"] = args.assets
    cmd = [sys.executable, "-m", "utils.sample"]
    if args.gpus > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.gpus}", "--module", "utils.sample"]
    cmd += ["--config", str(args.config.resolve()), "--device", args.device]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.checkpoint:
        cmd += ["--checkpoint", str(Path(args.checkpoint).resolve())]
    if args.resume:
        cmd += ["--resume"]
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
