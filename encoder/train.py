"""Shared FragHub pretraining and MSG/NPLIB1 fine-tuning."""
import argparse
import copy
import json
import math
import os
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .config import load, path
from .data import ARRAY_NAMES, Inputs, train_plan, monitor_indices, evaluate
from .memory import enable_activation_checkpointing
from .model import build, TensorForward, load_weights, finetune_loss
from .state import (atomic_json, append_json, export, get_rng, model_hash,
                    save, seed_all, set_rng, sha256)
from utils.artifacts import publish


def setup_runtime(cfg):
    runtime = cfg['runtime']
    if runtime['device'] == 'cpu':
        return torch.device('cpu'), []
    if runtime['device'] != 'cuda':
        raise ValueError('runtime.device must be cuda or cpu')
    ids = runtime['gpu_ids']
    if not torch.cuda.is_available() or max(ids) >= torch.cuda.device_count():
        raise RuntimeError(f'Requested GPUs {ids}; visible CUDA device count is {torch.cuda.device_count()}')
    torch.cuda.set_device(ids[0])
    return torch.device(f'cuda:{ids[0]}'), ids


def parallel_forward(model, gpu_ids):
    forward = TensorForward(model)
    if len(gpu_ids) > 1:
        forward = nn.DataParallel(forward, device_ids=gpu_ids, output_device=gpu_ids[0])
    return forward


def start_run(cfg, stage, resume, inputs):
    root = path(cfg['output_root']) / ('pretrain' if stage == 'pretrain' else cfg['dataset'] + '_finetune')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'run.json'
    recorded_cfg = copy.deepcopy(cfg)
    recorded_cfg['stage'] = stage
    record = {'stage': stage, 'config': recorded_cfg, 'inputs': inputs}
    if resume:
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != record:
            raise ValueError('Resume configuration or input hashes differ from the saved run')
        if not (root / 'resume.pth').exists():
            raise FileNotFoundError(root / 'resume.pth')
    else:
        if manifest.exists() or (root / 'resume.pth').exists():
            raise FileExistsError(f'Run already exists: {root}. Use --resume or a new output_root.')
        atomic_json(manifest, record)
    return root


def restore(root, model, optimizer, scheduler):
    # Resume files are generated locally by this trainer and include Python RNG state.
    saved = torch.load(root / 'resume.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    set_rng(saved['rng'])
    return saved


def resume_state(model, optimizer, scheduler, **progress):
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=get_rng(), **progress)


def run_pretrain(cfg, device, gpu_ids, resume=False):
    from .prepare import prepare_pretrain
    from .pretrain_data import load_data
    from .pretrain_math import batch_graphs_to_padded_data, compute_contrastive_logits, build_scheduler, evaluate_during_training
    settings = cfg['pretrain']
    prepared = prepare_pretrain(cfg)
    inputs = json.loads((prepared / 'PREPARED.json').read_text(encoding='utf-8'))
    weights = path(settings['weights'])
    if not resume and weights.exists():
        raise FileExistsError(f'Choose a new weights path; refusing to replace {weights}')
    seed_all(settings['seed'], cfg['runtime']['threads'])
    train_loader, val_loader, _, graphs = load_data(
        str(path(settings['parquet'])), str(prepared / 'clean_smiles_dict.pt'),
        'predefined', str(prepared / 'splits_clean_seed42.pkl'),
        batch_size=settings['batch_size'], ordered=True,
        ordered_sub_batch_size=settings['ordered_sub_batch_size'],
        random_seed=settings['seed'], device=str(device))
    model = build(settings['temperature']).to(device)
    initial_hash = model_hash(model)
    if settings['activation_checkpointing']:
        enable_activation_checkpointing(model)
    forward = parallel_forward(model, gpu_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'],
                                 betas=tuple(settings['betas']), eps=settings['eps'],
                                 weight_decay=settings['weight_decay'])
    scheduler = build_scheduler(optimizer, settings['epochs'] * len(train_loader),
                                scheduler_type='WarmCosine', warmsteps=settings['warmup_steps'])
    root = start_run(cfg, 'pretrain', resume, inputs)
    atomic_json(root / 'model.json', {'initial_model_sha256': initial_hash,
                                    'parameters': sum(p.numel() for p in model.parameters()),
                                    'batches_per_epoch': len(train_loader)})
    first_epoch, updates = 0, 0
    if resume:
        saved = restore(root, model, optimizer, scheduler)
        first_epoch, updates = saved['completed_epochs'], saved['updates']
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(first_epoch, settings['epochs']):
        losses = []
        for sos, formula, mask, smiles, _ in train_loader:
            optimizer.zero_grad()
            graph = batch_graphs_to_padded_data([graphs[s][0] for s in smiles], device=str(device))
            embeddings = forward(sos, formula, mask, graph.X, graph.E, graph.y, graph.node_mask)
            logits, reverse = compute_contrastive_logits(*embeddings, 1.0 / model.inv_temperature)
            labels = torch.arange(len(smiles), device=device)
            loss = (criterion(logits, labels) + criterion(reverse, labels)) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pretraining loss')
            loss.backward()
            optimizer.step()
            model.inv_temperature.data.clamp_(min=1.0 / 200.0)
            scheduler.step()
            updates += 1
            losses.append(loss.item())
            row = {'epoch': epoch + 1, 'step': updates, 'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}
            append_json(root / 'steps.jsonl', row)
            if updates % 10 == 0:
                print(json.dumps(row), flush=True)
        args = SimpleNamespace(mode='contrastive', device=str(device), temperature=settings['temperature'])
        metrics = evaluate_during_training(model, val_loader, graphs, args, epoch=epoch + 1)
        model.train()
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError('Nonfinite pretraining validation metric')
        append_json(root / 'metrics.jsonl', {'epoch': epoch + 1, 'step': updates, 'train_loss': float(np.mean(losses)), **metrics})
        save(root / 'resume.pth', resume_state(model, optimizer, scheduler, completed_epochs=epoch + 1, updates=updates))
        if (epoch + 1) % settings['save_every_epochs'] == 0:
            export(root / 'snapshots' / f'epoch{epoch + 1:04d}.pth', model)
    export(weights, model)
    publish(weights.parent / 'latest.json', weights, 'encoder')
    atomic_json(root / 'COMPLETED.json', {'stage': 'pretrain', 'epochs': settings['epochs'], 'updates': updates,
                                         'weights': str(weights), 'weights_sha256': sha256(weights),
                                         'model_sha256': model_hash(model)})
    print(f'Pretrained encoder: {weights}', flush=True)


def run_finetune(cfg, device, gpu_ids, resume=False, stop_after=None):
    settings = cfg['finetune']
    dataset = cfg['datasets'][cfg['dataset']]
    directory = path(dataset['prepared_dir'])
    initial = path(settings['initial_weights'])
    if not initial.exists():
        raise FileNotFoundError(f'Pretrained weights missing: {initial}. Run --stage pretrain first.')
    data = Inputs(directory, device)
    inputs = {name: sha256(directory / f'{name}.npy') for name in ARRAY_NAMES + ('train_indices', 'val_indices')}
    inputs['initial_weights'] = sha256(initial)
    seed_all(settings['seed'], cfg['runtime']['threads'], deterministic=True)
    train_ids, val_ids = data.indices('train'), data.indices('val')
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError('Train and val row indices overlap')
    model = build(30.0).to(device)
    load_weights(model, initial)
    initial_hash = model_hash(model)
    if settings['activation_checkpointing']:
        enable_activation_checkpointing(model)
    forward = parallel_forward(model, gpu_ids)
    # Model construction must not consume the training/dropout random stream.
    seed_all(settings['seed'], cfg['runtime']['threads'], deterministic=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                  weight_decay=settings['weight_decay'], betas=tuple(settings['betas']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=settings['scheduler_factor'],
        patience=settings['scheduler_patience'], min_lr=settings['min_lr'])
    plan = train_plan(train_ids, dataset['steps'], settings['batch_size'], settings['seed'])
    monitor = monitor_indices(val_ids, settings['monitor_ratio'], settings['seed'])
    weights_dir = path(dataset['weights_dir'])
    if not resume and weights_dir.exists() and any(weights_dir.glob('*.pth')):
        raise FileExistsError(f'Weights already exist in {weights_dir}; choose a new weights_dir')
    root = start_run(cfg, 'finetune', resume, inputs)
    np.save(root / 'train_plan.npy', plan)
    np.save(root / 'monitor_val.npy', monitor)
    atomic_json(root / 'model.json', {'initial_model_sha256': initial_hash,
                                    'train_plan_sha256': sha256(root / 'train_plan.npy'),
                                    'monitor_sha256': sha256(root / 'monitor_val.npy')})
    first, best, best_iteration = 0, float('inf'), None
    if resume:
        saved = restore(root, model, optimizer, scheduler)
        first, best, best_iteration = saved['updates'], saved['best_loss'], saved['best_iteration']
    limit = dataset['steps'] if stop_after is None else min(stop_after, dataset['steps'])
    if limit <= first:
        raise ValueError('Requested stop is at or before the saved update')
    model.train()
    for iteration in range(first, limit):
        ids = plan[iteration]
        ids = ids[ids >= 0]
        if iteration < settings['warmup_steps']:
            for group in optimizer.param_groups:
                group['lr'] = settings['lr'] * (iteration + 1) / settings['warmup_steps']
        optimizer.zero_grad(set_to_none=True)
        loss = finetune_loss(model, forward(*data.batch(ids)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at iteration {iteration}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'])
        if not torch.isfinite(grad):
            raise FloatingPointError(f'Nonfinite gradient at iteration {iteration}')
        optimizer.step()
        row = {'iteration': iteration, 'step': iteration + 1, 'loss': loss.item(),
               'grad_norm': grad.item(), 'lr': optimizer.param_groups[0]['lr']}
        append_json(root / 'steps.jsonl', row)
        if iteration % 25 == 0:
            print(json.dumps(row), flush=True)
        scheduled = iteration > 0 and iteration % settings['monitor_every'] == 0
        final = iteration == dataset['steps'] - 1
        # NPLIB1 includes the final update in its validation schedule.
        monitor_now = scheduled or (final and cfg['dataset'] == 'nplib1')
        if monitor_now or final or iteration == limit - 1:
            metrics = None
            if monitor_now:
                metrics = evaluate(model, TensorForward(model), data, monitor, settings['batch_size'])
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError('Nonfinite fine-tuning validation metric')
                scheduler.step(metrics['loss'])
                if metrics['loss'] < best:
                    best, best_iteration = metrics['loss'], iteration
                    export(weights_dir / 'best_loss.pth', model)
                append_json(root / 'metrics.jsonl', {'iteration': iteration, 'step': iteration + 1, **metrics})
            if scheduled or final:
                export(weights_dir / f'iter{iteration:05d}.pth', model)
            save(root / 'resume.pth', resume_state(model, optimizer, scheduler, updates=iteration + 1,
                                                  best_loss=best, best_iteration=best_iteration))
    if limit == dataset['steps']:
        final_path = weights_dir / f'final_step{limit}.pth'
        export(final_path, model)
        publish(weights_dir / 'latest.json', final_path, 'encoder', dataset=cfg['dataset'], updates=limit)
        atomic_json(root / 'COMPLETED.json', {'stage': 'finetune', 'dataset': cfg['dataset'], 'updates': limit,
                                             'weights': str(final_path), 'weights_sha256': sha256(final_path),
                                             'model_sha256': model_hash(model), 'best_iteration': best_iteration})
        print(f'Fine-tuned encoder: {final_path}', flush=True)
    else:
        print(f'Stopped after {limit} updates; resume with --resume.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/encoder.yml')
    parser.add_argument('--stage', choices=['pretrain', 'finetune', 'all'])
    parser.add_argument('--dataset', choices=['msg', 'nplib1'])
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--initial-weights', help='Override fine-tuning initialization, including when resuming a stage-all run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Stop fine-tuning after this absolute update; keep resume state')
    args = parser.parse_args()
    cfg = load(args.config, dataset=args.dataset, gpus=args.gpus, stage=args.stage)
    if args.initial_weights:
        cfg['finetune']['initial_weights'] = args.initial_weights
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if cfg['stage'] == 'all' and args.resume:
        parser.error('Resume a specific stage with --stage pretrain or --stage finetune')
    if args.stop_after is not None and (args.stop_after < 1 or cfg['stage'] == 'pretrain'):
        parser.error('--stop-after requires positive updates and fine-tuning')
    device, ids = setup_runtime(cfg)
    if cfg['stage'] in ('pretrain', 'all'):
        run_pretrain(cfg, device, ids, args.resume)
    if cfg['stage'] == 'all':
        cfg['finetune']['initial_weights'] = cfg['pretrain']['weights']
    if cfg['stage'] in ('finetune', 'all'):
        run_finetune(cfg, device, ids, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
