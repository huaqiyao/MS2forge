"""Read historical names while writing only canonical MS2Forge identifiers."""
from types import SimpleNamespace


CANDIDATE_CACHE_SCHEMA = "ms2forge_candidate_cache.v1"
# Read-only compatibility with existing candidate files; the payload is unchanged.
SUPPORTED_CANDIDATE_CACHE_SCHEMAS = frozenset({
    CANDIDATE_CACHE_SCHEMA, "flash_candidate_cache.v1",
})


def normalize_model_config(config):
    """Canonicalize the old model key without mutating caller/checkpoint data.

    Equal aliases are accepted; conflicting settings must never silently change
    inference or pass the exact-resume configuration check.
    """
    if config is None:
        return None
    result = dict(vars(config) if isinstance(config, SimpleNamespace) else config)
    # This spelling is retained only to read previously saved configurations.
    legacy_key = "flash"
    present = [key for key in ("ms2forge", legacy_key) if key in result]
    settings = []
    for key in present:
        value = result[key]
        settings.append(dict(vars(value) if isinstance(value, SimpleNamespace) else value))
    if len(settings) == 2 and settings[0] != settings[1]:
        raise ValueError("Conflicting current and legacy MS2Forge model settings")
    if settings:
        result.pop(legacy_key, None)
        result["ms2forge"] = settings[0]
    return result
