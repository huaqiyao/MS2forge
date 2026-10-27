"""Pure model exports and separate resumable training state."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(filename, obj):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(filename)


def save(filename, obj):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(filename)


def export(filename, model):
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise FloatingPointError('Cannot export nonfinite model weights')
    save(filename, state)


def get_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same visible GPU count')
        torch.cuda.set_rng_state_all(state['cuda'])


def seed_all(seed, threads=4, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def append_json(filename, row):
    with Path(filename).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, default=str) + '\n')
