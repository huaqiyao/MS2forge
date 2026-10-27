"""Launcher for EQ-COND Graph2Mol/MS2Mol and the encoder preparation tools."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from utils.config import external_root

REPO = Path(__file__).resolve().parent


def command(dataset, stage, root, gpus):
    if stage in ("align", "pretrain"):
        args = ["--stage", "finetune" if stage == "align" else "pretrain"]
        if dataset is not None:
            args += ["--dataset", dataset]
        if gpus is not None:
            args += ["--gpus", str(gpus)]
        return [sys.executable, "-m", "encoder.train"] + args
    elif stage in ("graph2mol", "ms2mol"):
        module = "utils.train"
        args = ["--config", f"configs/{stage}.yml"]
    else:
        module = "utils.sample"
        args = ["--config", "configs/sample.yml"]
    if dataset is not None:
        args += ["--dataset", dataset]
    prefix = [sys.executable, "-m", module]
    # Default to 8 GPUs; pass --gpus 1 to run on a single GPU.
    gpus = 8 if gpus is None else gpus
    if gpus > 1:
        prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                  f"--nproc_per_node={gpus}", "--module", module]
    return prefix + args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", nargs="?", choices=["msg", "nplib1"],
                   help="Optional override; otherwise use dataset in the YAML.")
    p.add_argument("stage", choices=["pretrain", "align", "graph2mol", "ms2mol", "sample"])
    p.add_argument("--assets", help="Absolute external directory; overrides MS2FORGE_ASSETS.")
    p.add_argument("--gpus", type=int,
                   help="Encoder: use YAML GPU IDs by default; decoder: default 8. Use --gpus 1 for a single GPU.")
    p.add_argument("--dry-run", action="store_true")
    # Arguments after -- are forwarded unchanged to the selected script.
    args, extra = p.parse_known_args()
    if args.assets:
        os.environ["MS2FORGE_ASSETS"] = args.assets
    if args.gpus is not None and args.gpus < 1:
        p.error("--gpus must be at least 1")
    try:
        root = None if args.stage in ("pretrain", "align") else external_root()
        cmd = command(args.dataset, args.stage, root, args.gpus)
    except ValueError as exc:
        p.error(str(exc))
    cmd += extra[1:] if extra[:1] == ["--"] else extra
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
