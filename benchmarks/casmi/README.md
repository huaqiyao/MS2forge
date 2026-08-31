# CASMI benchmark adapter

This directory contains code only:

- `casmi_dataset.py` builds the fixed heavy-atom composition from a molecular
  formula without using truth connectivity.
- `sample_casmi_deterministic.py` makes stochastic sampling stable per spectrum
  across distributed ranks and shard layouts.

The CASMI spectra, labels, subformula files, cohort manifests and caches are
datasets and are intentionally excluded. Supply a compatible dataset root and
sampling config separately. Run the wrapper from the repository root so that
`scripts/`, `utils/` and `models/` are importable.
