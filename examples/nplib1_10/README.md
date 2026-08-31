# NPLIB1 ten-spectrum demonstration

This bundle contains ten real `[M+H]+` spectra from the
`canopus_hplus_100_0` test split, their labels, peak-level subformula
annotations, and ten fp16 x 512 Zms vectors. Model checkpoints are not included.

## Selection protocol

Samples satisfy all of the following criteria:

- Membership in the official test split.
- Unique InChIKey.
- Supported molecular-formula elements.
- At least 40 positive-intensity peaks and 35 unique m/z values.
- At least 80% connected valid candidates in the frozen selection run.
- Exact Top-1 recovery in the frozen selection run.

Morgan-fingerprint diversity, spectrum quality, instrument metadata, and
candidate frequency were used for greedy selection. This is a curated
functional subset, not an unbiased benchmark sample.

## Offline validation

```bash
python examples/nplib1_10/run_demo.py
```

The command verifies SHA-256 values, parses all spectra and subformula files,
checks the Zms tensor shapes, and recomputes Top-k metrics from the frozen
candidate output. Expected results are:

```text
status: passed
spectra: 10
top1: 9/10
top5: 10/10
top10: 10/10
```

## Model inference

```bash
export MS2MOL_CKPT=/absolute/path/to/ms2mol_best.pt
export ALIGN_CKPT=/absolute/path/to/align_best.pt
bash examples/nplib1_10/run_inference.sh
```

The validated run used one GPU, 100 candidates per spectrum, 20 BFN steps,
real Zms and instrument conditions, and `diffms_2d` ranking. Runtime was 19.7
seconds. Top-1, Top-5, and Top-10 accuracy were 90%, 100%, and 100%; molecular
validity was 99.6%, and connected validity was 99.5%.

## Contents

```text
cache/zms_v1.pt          Ten Zms vectors
configs/sample.yml       Inference configuration
data/                    Spectra, labels, split, and subformula annotations
expected/                Frozen and rerun metrics
provenance.json          Dataset and artifact provenance
scripts/                 Selection and bundle-preparation scripts
selection.tsv            Human-readable selection table
selection.json           Machine-readable selection record
SHA256SUMS               File-integrity manifest
```
