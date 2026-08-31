# Code map

## Generation

- `models/model.py`: spectrum encoder, molecular encoder, and BFN model.
- `models/gnn.py`: graph-network backbone.
- `scripts/sample.py`: conditioned candidate generation and evaluation.
- `configs/sample.yml`: validated inference protocol.
- `utils/dataset.py`: MSG and SMILES datasets and Zms/Zmol caches.
- `utils/transforms.py`: graph features, batching, and cache injection.
- `utils/eval_utils.py`: exact, isomorphic, InChIKey, and DiffMS metrics.
- `utils/reconstruct.py`: RDKit molecule reconstruction.

## Training

- `scripts/train.py`: Graph2Mol, MS2Mol, and joint training stages.
- `configs/train_graph2mol.yml`: Graph2Mol configuration.
- `configs/train_ms2mol.yml`: MS2Mol configuration.
- `configs/train_joint.yml`: joint-training configuration.
- `scripts/train_*_ddp.sh`: distributed launchers.

## Reranking

- `scripts/export_rerank_candidates.py`: label-free generator-to-reranker export.
- `rerank/ms2forge_evidencerank_lite_features.py`: fragmentation evidence.
- `rerank/ms2forge_spectrum_structure_ranker.py`: neural candidate ranker.
- `rerank/ms2forge_apply_frozen_test.py`: frozen score fusion.
- `rerank/FROZEN_TEST_DECISION_20260721.json`: frozen protocol and hashes.

## Validation

- `run_test/demo_forward.py`: checkpoint-free forward pass.
- `scripts/verify_release.py`: artifact and state-dict validation.
- `scripts/reevaluate_candidate_cache.py`: CPU candidate-cache evaluation.
- `tests/test_release_contract.py`: configuration and loading contracts.
- `tests/test_export_rerank_candidates.py`: candidate-export contracts.
