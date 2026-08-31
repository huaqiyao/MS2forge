# EvidenceRank-SSR code

This directory contains the current reranking implementation and the frozen
validation-decision records, without model binaries or large candidate tables.

- `ms2forge_evidencerank_lite_features.py`: label-free fragmentation and
  peak-evidence features.
- `ms2forge_spectrum_structure_ranker.py`: SSR data preparation, training,
  candidate encoding and scoring.
- `ms2forge_apply_frozen_test.py`: frozen Stage-A/SSR fusion and blind-test
  inference.

The corresponding `.pkl`, `.pt` and multi-GB TSV artifacts belong under
`rerank/artifacts/` after separate download. Their provenance is recorded in
the repository metadata and frozen decision JSON. The generator bridge is
`scripts/export_rerank_candidates.py`.
