"""Verified pointers to explicitly published model checkpoints."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path, stage=None):
    path = Path(path).expanduser().resolve()
    if path.suffix != '.json':
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint missing: {path}')
        return path
    if not path.is_file():
        raise FileNotFoundError(f'No published checkpoint: {path}. Complete the preceding stage or supply an explicit checkpoint.')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != 'ms2forge.checkpoint_pointer.v1':
        raise ValueError(f'Not a checkpoint pointer: {path}')
    if stage and record.get('stage') != stage:
        raise ValueError(f'Expected {stage} checkpoint, got {record.get("stage")}')
    target = Path(record['path'])
    if not target.is_absolute():
        target = path.parent / target
    target = target.resolve()
    if target.suffix == '.json' or sha256(target) != record['sha256']:
        raise ValueError(f'Published checkpoint content changed: {target}')
    return target


def publish(pointer, checkpoint, stage, **metadata):
    pointer, checkpoint = Path(pointer).resolve(), Path(checkpoint).resolve()
    record = dict(schema='ms2forge.checkpoint_pointer.v1', path=str(checkpoint),
                  sha256=sha256(checkpoint), stage=stage, **metadata)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_name(pointer.name + '.tmp')
    tmp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    tmp.replace(pointer)
    return record


def encoder_identity(config):
    """An explicitly selected physical encoder supersedes old historical pins."""
    if config.get('encoder_checkpoint'):
        return sha256(resolve(config['encoder_checkpoint'], 'encoder'))
    return config.get('encoder_sha256')


def require_encoder(payload, expected, label):
    recorded = payload.get('encoder_sha256') or payload.get('config', {}).get('encoder_sha256')
    if expected and recorded != expected:
        raise ValueError(f'{label} encoder identity mismatch: checkpoint={recorded}, cache={expected}. '
                         'Use weights and caches from the same encoder; legacy weights without lineage need verified provenance.')
