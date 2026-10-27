"""Read YAML with explicit environment-variable expansion."""
import os
import re
from pathlib import Path
import yaml
from utils.compat import normalize_model_config


def external_root():
    value = os.environ.get("MS2FORGE_ASSETS")
    if not value:
        raise ValueError("Set MS2FORGE_ASSETS to an absolute directory outside this repository.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("MS2FORGE_ASSETS must be an absolute path.")
    repo = Path(__file__).resolve().parents[1]
    resolved = path.resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError("MS2FORGE_ASSETS must be outside this repository.")
    return resolved


def expand(value, variables=None):
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, str):
        for key, replacement in (variables or {}).items():
            value = value.replace("${" + key + "}", str(replacement))
        if "${MS2FORGE_ASSETS}" in value:
            value = value.replace("${MS2FORGE_ASSETS}", str(external_root()))
        value = os.path.expandvars(os.path.expanduser(value))
        if re.search(r"\$\{[^}]+\}", value):
            raise ValueError("Unresolved environment variable in config: " + value)
    return value


def load_yaml(stream, dataset=None):
    cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Expected a YAML configuration mapping.")
    selected = dataset if dataset is not None else cfg.get("dataset")
    if selected not in ("msg", "nplib1"):
        raise ValueError("dataset must be msg or nplib1")
    cfg["dataset"] = selected
    if "model" in cfg:
        cfg["model"] = normalize_model_config(cfg["model"])
    # A filename change does not change encoder identity. Keep the original
    # per-dataset checks when sharing a configuration across datasets.
    identities = cfg.pop("encoder_sha256_by_dataset", None)
    if identities is not None:
        if not isinstance(identities, dict) or selected not in identities:
            raise ValueError("Missing encoder identity for dataset: " + selected)
        cfg["encoder_sha256"] = identities[selected]
    return expand(cfg, {"DATASET": selected,
                        "REPO_ROOT": Path(__file__).resolve().parents[1].as_posix()})
