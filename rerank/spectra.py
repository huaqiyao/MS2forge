"""Shared MS/MS reader for MSG and NPLIB1."""
import math
from pathlib import Path


def read_spectrum(path):
    parent, ion, section = 0.0, '', None
    explicit, collision = [], []
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('>'):
            key, *value = line.split(maxsplit=1)
            key = key.lower()
            if key == '>parentmass':
                parent = float(value[0]) if value else 0.0
            elif key == '>ionization':
                ion = value[0].replace(' ', '') if value else ''
            if key == '>ms2peaks':
                section = explicit
            elif key.startswith('>collision'):
                section = collision
            else:
                section = None
            continue
        if section is not None:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f'{path}: malformed MS/MS peak: {line}')
            mz, intensity = map(float, fields[:2])
            if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                section.append((mz, intensity))
    peaks = explicit or collision
    if not peaks or not math.isfinite(parent) or parent <= 0:
        raise ValueError(f'{path}: positive precursor mass and nonempty MS/MS peaks required')
    if not explicit:
        merged = {}
        for mz, intensity in peaks:
            merged[mz] = merged.get(mz, 0.0) + intensity
        peaks = list(merged.items())
    return parent, ion, tuple(sorted(peaks, key=lambda p: -p[1]))
