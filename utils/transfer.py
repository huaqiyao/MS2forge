"""Transfer all Graph2Mol weights to an MS2Mol student."""
from typing import Any, Mapping

def load_graph_parent(student: Any, parent_state: Mapping[str, Any]) -> list[str]:
    """Copy the physical parent state through the standard module loader.

    The frozen helper's ``load_parent_state_dict`` intentionally filters
    ``formula_condition.*`` and ``layer_conditions.*`` from its parent-key
    set.  Calling it here would leave 50 learned condition tensors fresh even
    though the physical parent contains all 54.  This function uses the
    standard ``load_state_dict(..., strict=False)`` on the already constructed
    MS student, where the only allowed missing keys are the four new
    ``zms_adapter.*`` tensors.  Every one of the 371 parent tensors is checked
    for shape and exact post-copy equality.
    """

    student_state_before = dict(student.state_dict())
    parent_keys = set(parent_state)
    student_keys = set(student_state_before)
    if any(str(key).startswith("zms_adapter.") for key in parent_keys):
        raise ValueError("mature Graph parent unexpectedly contains a student zms_adapter key")
    allowed_missing = {
        key for key in student_keys if str(key).startswith("zms_adapter.")
    }
    if len(parent_keys) != 371:
        raise ValueError(f"mature Graph parent must contain 371 tensors, got {len(parent_keys)}")
    if len(allowed_missing) != 4:
        raise ValueError(f"MS student must expose exactly four fresh zms_adapter tensors, got {len(allowed_missing)}")
    expected_parent_keys = student_keys - allowed_missing
    if parent_keys != expected_parent_keys:
        missing_from_parent = sorted(expected_parent_keys - parent_keys)
        unexpected_parent = sorted(parent_keys - expected_parent_keys)
        raise ValueError(
            "mature Graph parent key set is incomplete or has unexpected keys: "
            f"missing={missing_from_parent[:8]} unexpected={unexpected_parent[:8]}"
        )
    condition_keys = {
        key for key in expected_parent_keys
        if str(key).startswith("formula_condition.") or str(key).startswith("layer_conditions.")
    }
    if len(condition_keys) != 54:
        raise ValueError(
            f"mature Graph parent must contain all 54 formula/FiLM tensors, got {len(condition_keys)}"
        )

    def shape(value: Any):
        value_shape = getattr(value, "shape", None)
        return tuple(int(item) for item in value_shape) if value_shape is not None else None

    for key in expected_parent_keys:
        expected_shape = shape(parent_state[key])
        student_shape = shape(student_state_before[key])
        if expected_shape is not None and student_shape is not None and expected_shape != student_shape:
            raise ValueError(f"mature parent/student tensor shape mismatch: {key}")

    load_state_dict = getattr(student, "load_state_dict", None)
    if not callable(load_state_dict):
        raise ValueError("MS student standard load_state_dict is unavailable")
    result = load_state_dict(parent_state, strict=False)
    if isinstance(result, Mapping):
        missing_keys = set(result.get("missing_keys", []))
        unexpected_keys = set(result.get("unexpected_keys", []))
    else:
        missing_keys = set(getattr(result, "missing_keys", []))
        unexpected_keys = set(getattr(result, "unexpected_keys", []))
    if missing_keys != allowed_missing or unexpected_keys:
        raise ValueError(
            "MS student parent load must preserve all mature Graph keys; "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}"
        )

    loaded_state = dict(student.state_dict())
    for key in expected_parent_keys:
        expected = parent_state[key]
        actual = loaded_state.get(key)
        if actual is None:
            raise ValueError(f"mature parent tensor was not loaded: {key}")
        try:
            expected_detached = expected.detach() if hasattr(expected, "detach") else expected
            actual_detached = actual.detach() if hasattr(actual, "detach") else actual
            if hasattr(expected_detached, "cpu"):
                expected_detached = expected_detached.cpu()
            if hasattr(actual_detached, "cpu"):
                actual_detached = actual_detached.cpu()
            equal = bool(expected_detached.equal(actual_detached)) if hasattr(expected_detached, "equal") else bool(expected_detached == actual_detached)
        except Exception as exc:
            raise ValueError(f"mature parent tensor equality check failed: {key}: {exc}") from exc
        if not equal:
            raise ValueError(f"mature parent tensor copy is not exact: {key}")
    return sorted(allowed_missing)
