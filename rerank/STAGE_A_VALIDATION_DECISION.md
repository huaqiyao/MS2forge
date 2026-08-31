# Stage A validation decision (not frozen; test sealed)

Date: 2026-07-19
Status: **provisional validation winner; not a final EvidenceRank release**

The six predeclared ER-Pareto variants were compared only on the canonical
validation split. `ER-SimRank-E8-A8` is the provisional Stage A winner because
it preserved/improved both exact-match counts and gave the best admissible
Tanimoto@1, with a directionally consistent official MCES result.

| Validation metric (n = 18,961) | Frozen EvidenceRank | ER-SimRank-E8-A8 | Delta |
|---|---:|---:|---:|
| Exact Top-1 count | 5,571 | 5,574 | +3 |
| Exact Top-1 accuracy | 29.3814% | 29.3972% | +0.0158 pp |
| Exact Top-10 count | 7,372 | 7,373 | +1 |
| Exact Top-10 accuracy | 38.8798% | 38.8851% | +0.0053 pp |
| Tanimoto@1 | 0.549234 | 0.549743 | +0.000509 |
| Tanimoto@10 | 0.633018 | 0.633451 | +0.000433 |
| MCES@1 | reference | reference − 0.008597 | lower is better |

Paired uncertainty:

- Tanimoto@1 delta 95% bootstrap CI: +0.000097 to +0.000915.
- Tanimoto@10 delta 95% bootstrap CI: +0.000275 to +0.000599.
- MCES@1 delta 95% bootstrap CI: −0.016508 to −0.000897.
- Exact Top-1 changes: 27 improvements versus 24 regressions; McNemar
  p = 0.7798. The exact-match count gain is therefore not statistically
  distinguishable from no change.

Interpretation: the evidence-only similarity head is saturated. It yields a
small, internally consistent structural-quality gain, but the effect size is
far too small to claim that the DualLGD Tanimoto gap has been closed. Because
the winner was selected among six variants on the same validation set, all
confidence intervals remain development evidence rather than confirmatory
test evidence.

Decision:

1. Keep frozen EvidenceRank intact as the reproducible paper baseline.
2. Carry `ER-SimRank-E8-A8` forward only as the Stage A base score for the
   train-only spectrum--structure neural reranker.
3. Keep the test split sealed until a single complete Stage B/RankAlign rule is
   selected and frozen.
4. Do not rename or overwrite EvidenceRank based on these validation results.
