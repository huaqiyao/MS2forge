"""Configuration for shared encoder training."""
import copy
import os
from pathlib import Path
import yaml
from utils.config import expand

REPO = Path(__file__).resolve().parents[1]


def path(value):
    result = Path(expand(str(value)))
    return result if result.is_absolute() else REPO / result


def load(filename, dataset=None, gpus=None, stage=None):
    with path(filename).open(encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    if dataset is not None:
        cfg['dataset'] = dataset
    if stage is not None:
        cfg['stage'] = stage
    if cfg['dataset'] not in ('msg', 'nplib1'):
        raise ValueError('dataset must be msg or nplib1')
    if cfg['stage'] not in ('pretrain', 'finetune', 'all'):
        raise ValueError('stage must be pretrain, finetune or all')
    if gpus is not None:
        if gpus < 1:
            raise ValueError('--gpus must be at least 1')
        cfg['runtime']['gpu_ids'] = list(range(gpus))
    ids = cfg['runtime']['gpu_ids']
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError('gpu_ids must contain distinct nonnegative integers')
    if cfg['finetune']['train_split'] != 'train' or cfg['finetune']['eval_split'] != 'val':
        raise ValueError('Fine-tuning uses train for updates and val for monitoring')
    for section in ('pretrain', 'finetune'):
        if cfg[section]['batch_size'] < 2:
            raise ValueError('Contrastive batch_size must be at least 2')
    return cfg
