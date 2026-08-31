# Spectrum--structure reranker: provisional validation decision

Status: **EXPERIMENTAL / NOT FROZEN / VALIDATION ONLY / TEST SEALED**

Date: 2026-07-19

## Predeclared selection

Three train-only objectives were compared after structure-disjoint epoch
monitoring: `PairOnly`, `Contrastive`, and `ContrastiveTan`. Each neural score
was group-normalized and fused with the Stage-A `ER-SimRank-E8-A8` score. A
candidate had to preserve both exact Top-1 and exact Top-10 validation counts;
among passing candidates the rule maximized Tanimoto@1, then Tanimoto@10.

The provisional winner is `SSR-ContrastiveTan` at a fixed fusion of 0.80
Stage-A score and 0.20 neural spectrum--structure score.

## Validation result (n = 18,961 spectra)

| Metric | Stage-A ER-SimRank-E8-A8 | SSR-ContrastiveTan | Paired delta | 95% paired-bootstrap CI |
|---|---:|---:|---:|---:|
| Exact Top-1 | 5,574 (29.3972%) | 5,625 (29.6662%) | +0.2690 pp | [+0.0738, +0.4641] pp |
| Exact Top-10 | 7,373 (38.8851%) | 7,376 (38.9009%) | +0.0158 pp | [-0.0369, +0.0633] pp |
| Tanimoto@1 | 0.549743 | 0.551515 | +0.001772 | [+0.000729, +0.002765] |
| Tanimoto@10 | 0.633451 | 0.634106 | +0.000655 | [+0.000270, +0.001040] |

Exact Top-1 has 211 paired improvements and 160 regressions (exact McNemar
`p = 0.00934`). Exact Top-10 has 14 improvements and 11 regressions
(`p = 0.690`), so it is non-inferior by the development count guard but not a
supported improvement claim.

Official MyopicMCES deltas are still being recomputed for every changed Top-1
or Top-10 set. The audit contains 2,750 changed Top-1 rankings, 11,659 changed
Top-10 sets, and 113,099 unique non-exact structure pairs.

## Interpretation boundary

This is a promising validation signal, not a final model. The same validation
split selected three objectives and their fusion grids, so the intervals are
paired uncertainty for the chosen comparison, not multiplicity-corrected
independent confirmation. No test score may be computed until the remaining
MCES audit is complete, the generator branch is adjudicated, one configuration
is frozen with hashes, and a single UEP test run is authorized.

Frozen EvidenceRank remains the official result throughout this development.
