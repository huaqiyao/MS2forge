# Reproducibility contract

## Stage dependencies

```text
alignment checkpoint
  |-- spectrum encoder  -> Zms cache -------------------+
  `-- molecule encoder  -> Zmol cache -----------+      |
                                                   v      v
pretraining SMILES -> Graph2Mol checkpoint -> MS2Mol checkpoint -> sampling
MSG spectra ------------------------------------------------------^
```

The Graph2Mol checkpoint initializes MS2Mol and serves as its knowledge-
distillation teacher. The MS2Mol checkpoint contains the Zms-to-Zmol adapter.
`scripts/sample.py` enables the adapter only when all four adapter parameters
are present.

## State-dict contracts

The historical Graph2Mol checkpoint contains four inactive edge-precision
parameters:

```text
edge_precision_head.0.weight
edge_precision_head.0.bias
edge_precision_head.2.weight
edge_precision_head.2.bias
```

These are the only unexpected keys accepted when loading Graph2Mol as the
MS2Mol initializer or teacher. Every other missing or unexpected key fails
release verification.

The released MS2Mol condition protocol is:

```yaml
student_condition_mode: real
teacher_condition_mode: real
default_instrument_idx: 2
default_ionization_idx: 0
```

The student uses Adapter(Zms). The teacher uses Zmol with the adapter disabled.

## Data identity

Exact reproduction uses the entry counts and SHA-256 values recorded in
`metadata/dataset_provenance.json`. Rebuilding HMDB, DSSTox, COCONUT, MOSES,
or MSG inputs with different source versions, ordering, or RDKit versions may
produce different byte-level datasets.

## Verification

```bash
python scripts/verify_release.py --skip-large-cache-hash
python -m pytest -q
```
