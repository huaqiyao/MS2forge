from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name):
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_stage_configs_are_explicit():
    expected = {
        "train_graph2mol.yml": ("graph2mol", 1460001),
        "train_ms2mol.yml": ("ms2mol", 80001),
        "train_joint.yml": ("joint", 100001),
    }
    for name, (stage, max_iters) in expected.items():
        config = load_yaml(name)
        assert config["model"]["stage"] == stage
        assert config["train"]["max_iters"] == max_iters
        assert config["dataset"]["cache_dir"] == "./data/cache"


def test_released_ms_protocol_is_real_conditioned():
    train = load_yaml("train_ms2mol.yml")["train"]
    sample = load_yaml("sample.yml")["evaluate"]
    assert train["student_condition_mode"] == "real"
    assert train["teacher_condition_mode"] == "real"
    assert sample["condition_mode"] == "real"


def test_no_lora_or_generic_freeze_controls_in_training_entry():
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8").lower()
    forbidden = (
        "class loralinear",
        "apply_lora_to_linears",
        "apply_finetune_controls",
        "frozen_parameter_digest",
    )
    for token in forbidden:
        assert token not in source


def test_rerank_bridge_and_dependency_are_declared():
    environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    pip_dependencies = next(
        item["pip"]
        for item in environment["dependencies"]
        if isinstance(item, dict) and "pip" in item
    )
    assert "lightgbm==4.6.0" in pip_dependencies
    assert (ROOT / "scripts" / "export_rerank_candidates.py").is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code_map = (ROOT / "docs" / "CODE_MAP.md").read_text(encoding="utf-8")
    assert "EvidenceRank-SSR" in readme
    assert "export_rerank_candidates.py" in readme
    assert "EvidenceRank-SSR implementation" in readme
    assert "ms2forge_spectrum_structure_ranker.py" in code_map
