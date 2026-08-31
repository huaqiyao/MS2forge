# MS2Forge

MS2Forge generates molecular structures from tandem mass spectra with a
Bayesian Flow Network (BFN) and ranks generated candidates using fragment
evidence and spectrum--structure compatibility.

![MS2Forge architecture](assets/ms2forge_architecture.png)

## Installation

```bash
conda env create -f environment.yml
conda activate ms2forge
```

The validated environment uses Python 3.9.23, PyTorch 2.3.1, CUDA 11.8,
RDKit 2024.09.4, and PyTorch Geometric 2.3.1. `environment_lock.yml` and
`requirements_freeze.txt` provide exact environment snapshots.

## Quick validation

```bash
python run_test/demo_forward.py --device auto
python -m pytest -q
```

## NPLIB1 demonstration

The repository includes ten NPLIB1 test spectra, peak-level subformula
annotations, a ten-entry Zms cache, and frozen candidate results.

```bash
python examples/nplib1_10/run_demo.py
```

Expected metrics are Top-1 9/10, Top-5 10/10, and Top-10 10/10. The subset is
a curated functional demonstration and is not an unbiased benchmark estimate.

To rerun model inference, provide the MS2Mol and alignment checkpoints:

```bash
export MS2MOL_CKPT=/absolute/path/to/ms2mol_best.pt
export ALIGN_CKPT=/absolute/path/to/align_best.pt
bash examples/nplib1_10/run_inference.sh
```

## Training

```bash
bash scripts/train_graph2mol_ddp.sh
bash scripts/train_ms2mol_ddp.sh
bash scripts/train_joint_ddp.sh
```

## Inference

```bash
python scripts/sample.py --config configs/sample.yml --batch_size 4
bash scripts/sample_ddp.sh
python scripts/export_rerank_candidates.py --help
```

The EvidenceRank-SSR implementation is located in `rerank/`.

## External artifacts

Training and full inference require separately distributed datasets, caches,
and checkpoints at the paths defined by the YAML configurations:

```text
checkpoints/align.pt
checkpoints/graph2mol_iter1460000.pt
checkpoints/ms2mol_iter80000.pt
data/cache/zms_v1.pt
data/cache/zmol_v1.pt
data/msg_diffms/
data/pretrain/pretrain_smiles.csv
rerank/artifacts/
```

Artifact hashes and evaluation protocols are stored in `metadata/`,
`evaluation/`, and `rerank/FROZEN_TEST_DECISION_20260721.json`.

## Repository structure

```text
benchmarks/   Benchmark adapters
configs/      Training and inference configurations
docs/         Code map and reproducibility contract
evaluation/   Frozen evaluation summaries
examples/     Executable real-data demonstration
metadata/     Dataset, cache, and checkpoint provenance
models/       MS2Forge and graph-network modules
rerank/       EvidenceRank-SSR implementation
run_test/     Checkpoint-free architecture test
scripts/      Training, sampling, export, and verification commands
tests/        Release-contract tests
utils/        Data, chemistry, reconstruction, and evaluation utilities
```

## Data attribution

NPLIB1, also known as the CANOPUS training dataset, was prepared from public
GNPS spectra. Publications using the demonstration data should cite GNPS and
CANOPUS:

- Wang, M. et al. *Nature Biotechnology* **34**, 828--837 (2016).
  https://doi.org/10.1038/nbt.3597
- Dührkop, K. et al. *Nature Biotechnology* **39**, 462--471 (2021).
  https://doi.org/10.1038/s41587-020-0740-8

## License

This project is released under the MIT License. See `LICENSE`.
