"""Shared scalar collision-energy parsing and 11-bin spectrum-token encoding."""
import math
from pathlib import Path
import re

ENERGY_ENCODING = 'scalar_ce_20unit_11bins_missing0_v1'
NCE_FIELDS = ('collisionenergynce', 'normalizedcollisionenergy', 'nce')
CE_FIELDS = ('collisionenergy', 'ce')
_SCALAR = re.compile(
    r'(?:(?:NCE|CE)\s*[:=]?\s*)?'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    r'\s*(?:%|eV|NCE|CE)?', re.IGNORECASE)


def _key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def scalar_energy(value):
    """Return a finite nonnegative scalar, or None for missing/invalid values."""
    if value is None or isinstance(value, bool):
        return None
    match = _SCALAR.fullmatch(str(value).strip())
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) and number >= 0 else None


def energy_bin(value):
    """Use intervals [0,20), ..., [180,200), [200,inf); missing maps to 0."""
    number = scalar_energy(value)
    return 0 if number is None else min(int(number / 20), 10)


def _values(row, fields):
    normalized = {_key(k): v for k, v in row.items()}
    for key in fields:
        number = scalar_energy(normalized.get(key))
        if number is not None:
            return number
    return None


def _spectrum_metadata(root, spec_id):
    """Read named CE fields from a SIRIUS .ms file without reading structure fields."""
    if Path(str(spec_id)).name != str(spec_id) or '/' in str(spec_id) or '\\' in str(spec_id):
        raise ValueError('Unsafe spectrum ID for collision-energy lookup')
    path = Path(root) / 'spec_files' / (str(spec_id) + '.ms')
    if not path.is_file():
        return {}
    values = {}
    with path.open(encoding='utf-8-sig') as handle:
        for line in handle:
            if not line.startswith(('>', '#')):
                continue
            match = re.match(r'^[>#]\s*([\w-]+)\s*(?:[:=]\s*|\s+)(.+?)\s*$', line)
            if match and _key(match[1]) in NCE_FIELDS + CE_FIELDS:
                key = _key(match[1])
                number = scalar_energy(match[2])
                if number is not None:
                    if key in values and values[key] != number:
                        raise ValueError(f'{spec_id}: multiple collision energies; provide a scalar in labels.tsv')
                    values[key] = number
    return values


def collision_energy_bin(row, root=None, spec_id=None):
    """Prefer NCE, then CE; fall back to .ms metadata when label energy is absent.

    Scalar CE is binned in its supplied units; no CE-to-NCE conversion is inferred.
    """
    value = _values(row, NCE_FIELDS)
    if value is None:
        value = _values(row, CE_FIELDS)
    if value is None and root is not None and spec_id is not None:
        fields = _spectrum_metadata(root, spec_id)
        value = _values(fields, NCE_FIELDS)
        if value is None:
            value = _values(fields, CE_FIELDS)
    return energy_bin(value)
