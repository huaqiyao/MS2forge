"""MS2Forge module."""

import os
import sys
import time
import argparse
import pickle
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import logging
from easydict import EasyDict
import yaml
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, FeaturizeMol2D
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError
from torch_geometric.transforms import Compose


def load_config(config_path):
    """load_config implementation."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed):
    """seed_all implementation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args):
    """setup_distributed implementation."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP training requires CUDA, but torch.cuda.is_available() is False')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=12))
        args.device = f'cuda:{local_rank}'
    return distributed, rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def ddp_barrier(distributed):
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


def wait_for_files(paths, logger, desc, poll_sec=30):
    """wait_for_files implementation."""
    paths = list(paths)
    while not all(os.path.exists(p) for p in paths):
        missing = [p for p in paths if not os.path.exists(p)]
        logger.warning(f'{desc}: waiting for filegenerate, missing {missing[:3]}')
        time.sleep(poll_sec)


def expected_smiles_processed_paths(dataset_cfg, atomic_numbers):
    root = cfg_get(dataset_cfg, 'root', './data/pretrain')
    smiles_file = cfg_get(dataset_cfg, 'smiles_file', os.path.join(root, 'pretrain_smiles.csv'))
    max_atoms = cfg_get(dataset_cfg, 'max_atoms', None)
    data_subset_ratio = float(cfg_get(dataset_cfg, 'data_subset_ratio', 1.0))
    base = os.path.splitext(os.path.basename(smiles_file))[0]
    suffix = f"smiles_{base}_max{max_atoms if max_atoms is not None else 'inf'}_atoms{len(list(atomic_numbers))}"
    if data_subset_ratio < 1.0:
        suffix += f"_{int(data_subset_ratio * 100)}pct"
    processed_path = os.path.join(root, f'processed_{suffix}.lmdb')
    keys_path = processed_path.replace('.lmdb', '_keys.pt')
    return processed_path, keys_path


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_RELEASE_ADAPTER_KEYS = frozenset({
    'zms_adapter.0.weight',
    'zms_adapter.0.bias',
    'zms_adapter.3.weight',
    'zms_adapter.3.bias',
})
_RELEASE_LEGACY_EDGE_PRECISION_KEYS = frozenset({
    'edge_precision_head.0.weight',
    'edge_precision_head.0.bias',
    'edge_precision_head.2.weight',
    'edge_precision_head.2.bias',
})


def assert_release_checkpoint_keys(missing, unexpected, context,
                                   allow_graph_to_ms=False,
                                   allow_graph_resume=False):
    """Reject every non-strict checkpoint mismatch outside the release contract."""
    actual = (frozenset(missing), frozenset(unexpected))
    allowed = {(frozenset(), frozenset())}
    if allow_graph_to_ms:
        allowed.add((_RELEASE_ADAPTER_KEYS, _RELEASE_LEGACY_EDGE_PRECISION_KEYS))
    if allow_graph_resume:
        allowed.add((frozenset(), _RELEASE_LEGACY_EDGE_PRECISION_KEYS))
    if actual not in allowed:
        raise RuntimeError(
            f'{context} checkpoint key set does not satisfy  release compatibility contract: '
            f'missing={sorted(actual[0])}, unexpected={sorted(actual[1])}'
        )


def resolve_condition_indices(batch, batch_size, device, config,
                              force_mode=None):
    """Resolve BFN condition labels without silently mixing protocols.

    Stage-II uses index 2 for instrument=NONE and index 0 for
    ionization=[M+H]+.  ``force_mode`` is used by the fixed teacher; otherwise
    the student follows ``train.student_condition_mode``.
    """
    mode = force_mode or cfg_get(config.train, 'student_condition_mode', 'real')
    if mode not in ('real', 'default'):
        raise ValueError(f'student_condition_mode must be one of  real/default, ; received  {mode!r}')
    if mode == 'default':
        instrument_default = int(cfg_get(config.train, 'default_instrument_idx', 2))
        ionization_default = int(cfg_get(config.train, 'default_ionization_idx', 0))
        return (
            torch.full((batch_size,), instrument_default, dtype=torch.long, device=device),
            torch.full((batch_size,), ionization_default, dtype=torch.long, device=device),
        )
    instrument = (
        batch.instrument_type_idx_batch.to(device)
        if getattr(batch, 'instrument_type_idx_batch', None) is not None
        else torch.zeros(batch_size, dtype=torch.long, device=device)
    )
    ionization = (
        batch.ionization_type_idx_batch.to(device)
        if getattr(batch, 'ionization_type_idx_batch', None) is not None
        else torch.zeros(batch_size, dtype=torch.long, device=device)
    )
    return instrument, ionization


def limit_dataset(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


def get_logger(name, log_dir=None):
    """get_logger implementation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_model(config, num_node_types, num_edge_types, atomic_numbers):
    """create_model implementation."""
    from models.model import FLASH
    model = FLASH(config.model, num_node_types, num_edge_types, atomic_numbers)
    return model, None


def _permute_nodes_within_batch(node_type, halfedge_index, batch_node, device):
    """_permute_nodes_within_batch implementation."""

    perm_map = torch.arange(node_type.size(0), device=device)
    n_graphs = int(batch_node.max().item()) + 1
    for g in range(n_graphs):
        mask = (batch_node == g).nonzero(as_tuple=False).view(-1)
        if mask.numel() <= 1:
            continue
        shuffled = mask[torch.randperm(mask.numel(), device=device)]
        perm_map[mask] = shuffled





    new_node_type = torch.empty_like(node_type)
    new_node_type[perm_map] = node_type


    new_halfedge_index = perm_map[halfedge_index]

    return new_node_type, new_halfedge_index, batch_node



def train_flash(model, batch, device, config, iteration=None, logger=None):
    """train_flash implementation."""
    batch = batch.to(device)
    model_core = unwrap_model(model)
    stage = getattr(config.model, 'stage', None)
    if stage not in ('graph2mol', 'ms2mol', 'joint'):
        raise ValueError(f"model.stage must be one of  graph2mol/ms2mol/joint, ; received  {stage!r}")

    # ============================================================


    # ============================================================
    if not hasattr(batch, 'cond_emb_cached') or batch.cond_emb_cached is None:
        raise ValueError(f"{stage} stage batch missing cond_emb_cached(from  align ckpt generated offline  cache)")

    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.node_type_batch.max().item() + 1)

    instrument_type_idx, ionization_type_idx = resolve_condition_indices(
        batch, batch_size, device, config
    )


    t = torch.rand(batch_size, device=device)
    edge_types_true = batch.halfedge_type
    theta = model_core.discrete_bayesian_update(
        t, edge_types_true, batch.halfedge_type_batch
    )

    if iteration is not None and iteration < 10 and logger is not None:
        logger.info(f"[Iter {iteration}] {stage} BFN: "
                    f"t∈[{t.min().item():.3f}, {t.max().item():.3f}], "
                    f"theta entropy={-(theta * theta.clamp(min=1e-8).log()).sum(-1).mean().item():.4f}")


    e_hat = model(
        node_types=batch.node_type,
        edge_index=batch.halfedge_index,
        batch_node=batch.node_type_batch,
        batch_edge=batch.halfedge_type_batch,
        instrument_type_idx=instrument_type_idx,
        ionization_type_idx=ionization_type_idx,
        t=t,
        edge_types_t=torch.zeros_like(edge_types_true),
        edge_theta=theta,
        cond_emb_cached=batch.cond_emb_cached,
    )

    losses = model_core.compute_bfn_loss(
        e_hat, edge_types_true, t, batch.halfedge_type_batch
    )
    total_loss = losses['total']

    loss_dict = {k: v for k, v in losses.items()}
    loss_dict['loss'] = total_loss
    loss_dict['_edge_types_true'] = edge_types_true.detach()
    loss_dict['_batch_edge'] = batch.halfedge_type_batch.detach()
    loss_dict['_edge_logits'] = e_hat
    loss_dict['_edge_raw_logits'] = getattr(model_core, '_last_x0_logits', None)
    loss_dict['_t'] = t.detach()
    loss_dict['_theta'] = theta.detach()
    return loss_dict


def print_first_iter_debug(batch, config, logger):
    """print_first_iter_debug implementation."""
    import random as rand

    logger.info("=" * 60)
    logger.info("[DEBUG] First-iteration diagnostics")
    logger.info("=" * 60)

    has_spectrum = hasattr(batch, 'batch_has_spectrum') and batch.batch_has_spectrum
    has_mask = hasattr(batch, 'has_spectrum_mask') and batch.has_spectrum_mask.any()
    has_embedding = hasattr(batch, 'pretrained_embedding_batch') and batch.pretrained_embedding_batch is not None

    logger.info(f"[Spectrum-feature check]")
    logger.info(f"  - batch_has_spectrum: {has_spectrum}")
    logger.info(f"  - has_spectrum_mask: {has_mask}")
    logger.info(f"  - pretrained_embedding_batch available: {has_embedding}")

    if has_embedding:
        emb = batch.pretrained_embedding_batch
        logger.info(f"  - pretrained_embedding_batch shape: {emb.shape}")
        logger.info(f"  - pretrained_embedding_batch range: [{emb.min().item():.4f}, {emb.max().item():.4f}]")

    logger.info("[Molecular formula and node-type check]")
    logger.info(f"  - node_type shape: {batch.node_type.shape}")
    logger.info(f"  - node_type unique values: {batch.node_type.unique().tolist()}")
    logger.info(f"  - Molecule count: {batch.num_graphs}")

    logger.info("[Training-label check]")
    logger.info(f"  - halfedge_type shape: {batch.halfedge_type.shape}")
    logger.info(f"  - halfedge_type unique values: {batch.halfedge_type.unique().tolist()}")

    mol_idx = rand.randint(0, batch.num_graphs - 1)
    logger.info(f"[Random molecule #{mol_idx} details]")

    mol_node_mask = (batch.node_type_batch == mol_idx)
    mol_nodes = batch.node_type[mol_node_mask]

    atomic_numbers = config.chem.atomic_numbers
    atom_symbols = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}

    formula_counts = {}
    for node_type_idx in mol_nodes.tolist():
        if node_type_idx < len(atomic_numbers):
            atom_num = atomic_numbers[node_type_idx]
            symbol = atom_symbols.get(atom_num, f'?{atom_num}')
            formula_counts[symbol] = formula_counts.get(symbol, 0) + 1

    formula_str = ''.join([f"{sym}{cnt}" if cnt > 1 else sym for sym, cnt in sorted(formula_counts.items())])
    logger.info(f"  - Molecular formula: {formula_str}")
    logger.info(f"  - Atom count: {mol_nodes.shape[0]}")

    logger.info("=" * 60)


def evaluate_reconstruction(model, val_dataset, device, config, mode, atomic_numbers, logger,
                            collate_fn, num_samples=100, distributed=False, rank=0,
                            world_size=1, eval_iteration=None):
    """evaluate_reconstruction implementation."""
    model = unwrap_model(model)
    model.eval()
    stage = getattr(config.model, 'stage', None)
    log_enabled = (not distributed) or rank == 0


    eval_batch_size = config.train.get('eval_batch_size', 4)
    if distributed:
        local_indices = list(range(rank, len(val_dataset), world_size))
        eval_dataset = torch.utils.data.Subset(val_dataset, local_indices)
    else:
        eval_dataset = val_dataset
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ============================================================
    # ============================================================

    # ============================================================
    if eval_iteration is not None:
        eval_seed = int(config.train.get('seed', 0)) + int(eval_iteration) * 1009 + rank * 1000003
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
    if log_enabled:
        logger.info(
            f'Evaluation configuration: eval_batch_size={eval_batch_size}, num_samples={num_samples}, '
            f'total samples={len(val_dataset)}, local-rank samples={len(eval_dataset)}, '
            f'world_size={world_size if distributed else 1}'
        )

    total_correct = 0
    total_edges = 0
    bond_correct = 0
    total_bonds = 0
    mol_exact_match = 0
    total_mols = 0


    top1_correct_count = 0
    top10_correct_count = 0

    with torch.no_grad():
        batch_pbar = tqdm(
            eval_loader,
            desc='Evaluating validation set' if not distributed else f'Evaluating validation set rank{rank}',
            leave=False,
            position=0,
            disable=distributed and rank != 0,
        )
        for batch_idx, batch in enumerate(batch_pbar):
            if batch is None:
                continue
            batch = batch.to(device)



            if not hasattr(batch, 'cond_emb_cached') or batch.cond_emb_cached is None:
                if log_enabled:
                    logger.warning("Evaluation batch has no cond_emb_cached tensor; skipping")
                continue
            batch_size_orig = batch.cond_emb_cached.size(0)

            instrument_type_idx, ionization_type_idx = resolve_condition_indices(
                batch, batch_size_orig, device, config
            )

            edge_types_true = batch.halfedge_type
            num_nodes = batch.node_type.shape[0]
            num_edges = batch.halfedge_index.shape[1]


            node_types_expanded = batch.node_type.repeat(num_samples)
            edge_index_expanded = batch.halfedge_index.repeat(1, num_samples)
            for i in range(1, num_samples):
                edge_index_expanded[:, i*num_edges:(i+1)*num_edges] += i * num_nodes

            batch_node_expanded = torch.cat([batch.node_type_batch + i * batch_size_orig for i in range(num_samples)])
            batch_edge_expanded = torch.cat([batch.halfedge_type_batch + i * batch_size_orig for i in range(num_samples)])

            instrument_type_idx_expanded = instrument_type_idx.repeat(num_samples)
            ionization_type_idx_expanded = ionization_type_idx.repeat(num_samples)

            sample_extra = {
                'cond_emb_cached': batch.cond_emb_cached.to(device).repeat(num_samples, 1),
            }

            pred_edge_types_all = model.sample_bfn(
                node_types=node_types_expanded,
                edge_index=edge_index_expanded,
                batch_node=batch_node_expanded,
                batch_edge=batch_edge_expanded,
                instrument_type_idx=instrument_type_idx_expanded,
                ionization_type_idx=ionization_type_idx_expanded,
                n_timesteps=config.model.flow.get('eval_n_timesteps', 100),
                disable_tqdm=False,
                **sample_extra,
            )


            all_predictions = []
            for i in range(num_samples):
                pred_edge_types = pred_edge_types_all[i*num_edges:(i+1)*num_edges]
                all_predictions.append(pred_edge_types)


            pred_edge_types = all_predictions[0]


            total_correct += (pred_edge_types == edge_types_true).sum().item()
            total_edges += edge_types_true.numel()


            bond_mask = edge_types_true > 0
            bond_correct += ((pred_edge_types == edge_types_true) & bond_mask).sum().item()
            total_bonds += bond_mask.sum().item()


            eval_mode = config.train.get('eval_mode', 'strict')
            num_mols_in_batch = batch.halfedge_type_batch.max().item() + 1
            chem_eval = eval_mode in ('isomorphic', 'inchikey', 'diffms_inchi', 'diffms_2d')

            if chem_eval and batch_idx == 0:
                if log_enabled:
                    metric_name_map = {
                        'inchikey': 'InChIKey',
                        'diffms_inchi': 'DiffMS valid-connected InChI',
                        'diffms_2d': 'DiffMS valid-connected non-isomeric canonical SMILES',
                    }
                    metric_name = metric_name_map.get(eval_mode, 'RDKit canonical SMILES')
                    logger.info(f'   eval_mode={eval_mode}: using  {metric_name} comparison '
                                f'(this timesevaluation {len(val_dataset)}  molecules; ground truth uses  batch.smiles convert directly, '
                                f'one predictive reconstruction pass Mol, total  ~{len(val_dataset)} times RDKit reconstruction)')
            n_iso_done_this_batch = 0
            n_iso_sanitize_fail = 0
            iso_t0 = time.time() if chem_eval else None
            for mol_idx in range(num_mols_in_batch):
                mol_mask = batch.halfedge_type_batch == mol_idx
                if mol_mask.sum() > 0:
                    mol_pred = pred_edge_types[mol_mask]
                    mol_true = edge_types_true[mol_mask]

                    if chem_eval:
                        from utils.eval_utils import (
                            edges_to_canonical_smiles,
                            edges_to_diffms_2d,
                            edges_to_diffms_inchi,
                            edges_to_inchikey,
                            mol_to_diffms_2d,
                        )
                        node_mask = batch.node_type_batch == mol_idx
                        nt = batch.node_type[node_mask].cpu()

                        he_full = batch.halfedge_index[:, mol_mask].cpu()
                        node_off = node_mask.nonzero(as_tuple=True)[0][0].item()
                        he_local = he_full - node_off
                        if eval_mode == 'inchikey':
                            pred_repr = edges_to_inchikey(nt, he_local, mol_pred.cpu(), atomic_numbers)
                        elif eval_mode == 'diffms_inchi':
                            pred_repr = edges_to_diffms_inchi(nt, he_local, mol_pred.cpu(), atomic_numbers)
                        elif eval_mode == 'diffms_2d':
                            pred_repr = edges_to_diffms_2d(nt, he_local, mol_pred.cpu(), atomic_numbers)
                        else:
                            pred_repr = edges_to_canonical_smiles(nt, he_local, mol_pred.cpu(), atomic_numbers)

                        from rdkit import Chem
                        true_raw = batch.smiles[mol_idx] if hasattr(batch, 'smiles') else None
                        if true_raw:
                            try:
                                _m = Chem.MolFromSmiles(true_raw)
                                if eval_mode == 'inchikey':
                                    true_repr = Chem.MolToInchiKey(_m) if _m else None
                                elif eval_mode == 'diffms_inchi':
                                    true_repr = Chem.MolToInchi(_m) if _m else None
                                elif eval_mode == 'diffms_2d':
                                    true_repr = mol_to_diffms_2d(_m)
                                else:
                                    true_repr = Chem.MolToSmiles(_m, canonical=True) if _m else None
                            except Exception:
                                true_repr = None
                        else:

                            if eval_mode == 'inchikey':
                                true_repr = edges_to_inchikey(nt, he_local, mol_true.cpu(), atomic_numbers)
                            elif eval_mode == 'diffms_inchi':
                                true_repr = edges_to_diffms_inchi(nt, he_local, mol_true.cpu(), atomic_numbers)
                            elif eval_mode == 'diffms_2d':
                                true_repr = edges_to_diffms_2d(nt, he_local, mol_true.cpu(), atomic_numbers)
                            else:
                                true_repr = edges_to_canonical_smiles(nt, he_local, mol_true.cpu(), atomic_numbers)
                        is_match = (pred_repr is not None and true_repr is not None and pred_repr == true_repr)
                        n_iso_done_this_batch += 1
                        if pred_repr is None:
                            n_iso_sanitize_fail += 1
                    else:
                        is_match = (mol_pred == mol_true).all().item()

                    if is_match:
                        mol_exact_match += 1


                    if num_samples > 1:

                        mol_predictions_gpu = []
                        for pred_single in all_predictions:
                            mol_pred_single = pred_single[mol_mask]
                            mol_predictions_gpu.append(mol_pred_single)


                        unique_predictions = []
                        unique_counts = []

                        for pred in mol_predictions_gpu:

                            found = False
                            for i, unique_pred in enumerate(unique_predictions):
                                if torch.equal(pred, unique_pred):
                                    unique_counts[i] += 1
                                    found = True
                                    break
                            if not found:
                                unique_predictions.append(pred)
                                unique_counts.append(1)


                        sorted_indices = sorted(range(len(unique_counts)), key=lambda i: unique_counts[i], reverse=True)


                        n_unique_predictions = len(unique_predictions)
                        top1_frequency = unique_counts[sorted_indices[0]] if len(sorted_indices) > 0 else 0


                        top1_match = False
                        top10_match = False

                        if len(sorted_indices) > 0:

                            if top1_frequency == 1:

                                for unique_pred in unique_predictions:
                                    if torch.equal(unique_pred, mol_true):
                                        top1_match = True
                                        break
                            else:

                                top1_match = torch.equal(unique_predictions[sorted_indices[0]], mol_true)


                            if n_unique_predictions >= 10 and top1_frequency == 1:

                                for unique_pred in unique_predictions:
                                    if torch.equal(unique_pred, mol_true):
                                        top10_match = True
                                        break
                            else:

                                for idx in sorted_indices[:min(10, len(sorted_indices))]:
                                    if torch.equal(unique_predictions[idx], mol_true):
                                        top10_match = True
                                        break


                        if top1_match:
                            top1_correct_count += 1
                        if top10_match:
                            top10_correct_count += 1

                    total_mols += 1


            if chem_eval and log_enabled:
                iso_dt = time.time() - iso_t0 if iso_t0 else 0
                logger.info(f"  [batch {batch_idx+1}/{len(eval_loader)}] "
                            f"RDKit reconstructed {n_iso_done_this_batch} molecules ({iso_dt:.2f}s; "
                            f"sanitize failures={n_iso_sanitize_fail}); cumulative={total_mols}")
                batch_pbar.set_postfix({
                    'molecule': total_mols,
                    'RDKit failures': f'{n_iso_sanitize_fail}/{n_iso_done_this_batch}',
                    f'{iso_dt:.1f}s/batch': '',
                })

    if distributed:
        metrics_tensor = torch.tensor([
            float(total_correct),
            float(total_edges),
            float(bond_correct),
            float(total_bonds),
            float(mol_exact_match),
            float(total_mols),
            float(top1_correct_count),
            float(top10_correct_count),
        ], dtype=torch.float64, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_correct = int(metrics_tensor[0].item())
        total_edges = int(metrics_tensor[1].item())
        bond_correct = int(metrics_tensor[2].item())
        total_bonds = int(metrics_tensor[3].item())
        mol_exact_match = int(metrics_tensor[4].item())
        total_mols = int(metrics_tensor[5].item())
        top1_correct_count = int(metrics_tensor[6].item())
        top10_correct_count = int(metrics_tensor[7].item())


    total_acc = total_correct / total_edges if total_edges > 0 else 0
    bond_acc = bond_correct / total_bonds if total_bonds > 0 else 0
    mol_acc = mol_exact_match / total_mols if total_mols > 0 else 0

    if log_enabled:
        logger.info(f"[evaluation] validationset statistics:")
        logger.info(f"  total edges: {total_edges}, totalmoleculecount: {total_mols}")
        logger.info(f"  TotalAcc: {total_acc:.4f}")
        logger.info(f"  BondAcc: {bond_acc:.4f}")
        logger.info(f"  MolAcc: {mol_exact_match}/{total_mols} = {mol_acc:.4f}")


    if num_samples > 1:
        top1_acc = top1_correct_count / total_mols if total_mols > 0 else 0.0
        top10_acc = top10_correct_count / total_mols if total_mols > 0 else 0.0
        if log_enabled:
            logger.info(f"  Top-1 MolAcc: {top1_acc:.4f} ({top1_correct_count}/{total_mols})")
            logger.info(f"  Top-10 MolAcc: {top10_acc:.4f} ({top10_correct_count}/{total_mols})")

    result_dict = {
        'recon_success_rate': 1.0,
        'edge_accuracy': total_acc,
        'bond_accuracy': bond_acc,
        'mol_accuracy': mol_acc
    }


    if num_samples > 1:
        result_dict['top1_mol_accuracy'] = top1_correct_count / total_mols if total_mols > 0 else 0.0
        result_dict['top10_mol_accuracy'] = top10_correct_count / total_mols if total_mols > 0 else 0.0

    return result_dict


def check_prediction_distribution(edge_logits, edge_types, logger, iteration):
    """check_prediction_distribution implementation."""
    with torch.no_grad():
        pred = torch.argmax(edge_logits, dim=-1)
        pred_dist = torch.bincount(pred, minlength=5)
        true_dist = torch.bincount(edge_types, minlength=5)


        total_acc = (pred == edge_types).float().mean().item()


        bond_mask = edge_types > 0
        if bond_mask.sum() > 0:
            bond_acc = ((pred == edge_types) & bond_mask).sum().item() / bond_mask.sum().item()
        else:
            bond_acc = 0.0


        no_bond_ratio = pred_dist[0].item() / pred.shape[0]

        logger.info(f"[distributioncheck] Iter {iteration}:")
        logger.info(f"  predicted distribution: {pred_dist.tolist()} (no-bond ratio: {no_bond_ratio:.2%})")
        logger.info(f"  true distribution: {true_dist.tolist()}")
        logger.info(f"  overall accuracy: {total_acc:.4f}, bond accuracy: {bond_acc:.4f}")

        if no_bond_ratio > 0.99:
            logger.warning(f"The model predicts almost exclusively no-bond classes ({no_bond_ratio:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_ms2mol.yml', help='Configuration-file path')
    parser.add_argument('--device', type=str, default='auto', help='Training device: auto/cpu/cuda/cuda:0')
    parser.add_argument('--logdir', type=str, default='./checkpoints/logs', help='Log directory')
    parser.add_argument('--ckptdir', type=str, default='./checkpoints/', help='Checkpoint output directory')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the latest checkpoint for the current mode')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='Previous-stage checkpoint (model weights only, strict=False). '
                             'Use an alignment checkpoint for Graph2Mol or a Graph2Mol checkpoint for MS2Mol.')
    parser.add_argument('--align_ckpt', type=str, default=None,
                        help='Alignment checkpoint containing ms_encoder and graph_encoder weights; '
                             'used only to initialize the MS2Mol spectrum encoder')
    parser.add_argument('--overfit_test', default=False, help='Run a single-batch overfitting test')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum number of training samples for a small validation run')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Maximum number of validation samples for a small validation run')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override config.train.max_iters')
    parser.add_argument('--val_freq', type=int, default=None,
                        help='override config.train.val_freq')
    parser.add_argument('--save_freq', type=int, default=None,
                        help='override config.train.save_freq')
    parser.add_argument('--optimizer_lr', type=float, default=None,
                        help='Override config.train.optimizer.lr')
    args = parser.parse_args()
    distributed, rank, local_rank, world_size = setup_distributed(args)
    is_main = is_main_process(rank)

    config = load_config(args.config)
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
    if args.save_freq is not None:
        config.train.save_freq = args.save_freq
    if args.optimizer_lr is not None:
        config.train.optimizer.lr = args.optimizer_lr
    seed_all(int(config.train.seed) + rank)

    if not distributed:
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif args.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"[WARNING] --device {args.device}  but  CUDA is unavailable, switching to  cpu")
            args.device = 'cpu'

    mode = 'flash-model'


    stage = getattr(config.model, 'stage', None)
    if stage not in ('graph2mol', 'ms2mol', 'joint'):
        raise ValueError(f"model.stage must be one of  graph2mol/ms2mol/joint, ; received  {stage!r}")






    if stage == 'graph2mol':

        dataset_cfg = config.dataset
        pretrain_root = cfg_get(dataset_cfg, 'root', './data/pretrain')
        config.dataset = EasyDict({
            'name': 'smiles',
            'root': pretrain_root,
            'smiles_file': cfg_get(
                dataset_cfg, 'smiles_file',
                os.path.join(pretrain_root, 'pretrain_smiles.csv'),
            ),
            'max_atoms': cfg_get(dataset_cfg, 'max_atoms', None),
            'split_seed': int(cfg_get(dataset_cfg, 'split_seed', 2026)),
            'split_ratio': cfg_get(dataset_cfg, 'split_ratio', [0.95, 0.025, 0.025]),
            'data_subset_ratio': float(cfg_get(dataset_cfg, 'data_subset_ratio', 1.0)),
            'cache_dir': cfg_get(dataset_cfg, 'cache_dir', './data/cache'),
            'atomic_numbers': list(config.chem.atomic_numbers)
                if isinstance(config.chem.atomic_numbers, (list, tuple))
                else [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53],
        })
    else:

        config.dataset = EasyDict({
            'name': 'msg_diffms',
            'root': config.dataset.get('root', './data/msg_diffms'),
            'data_split_mode': config.dataset.get('data_split_mode', 'split'),
            'instrument_type': config.dataset.get('instrument_type', 'all'),
            'data_subset_ratio': 1.0,
            'max_peaks': config.dataset.get('max_peaks', 128),
            'cache_dir': config.dataset.get('cache_dir', './data/cache'),


            'graph2mol': config.dataset.get('graph2mol', None),
            'graph2mol_root': config.dataset.get(
                'graph2mol_root',
                config.dataset.get('pretrain_root', './data/pretrain'),
            ),
            'graph2mol_smiles_file': config.dataset.get(
                'graph2mol_smiles_file',
                config.dataset.get('pretrain_smiles_file', None),
            ),
            'pretrain_root': config.dataset.get('pretrain_root', './data/pretrain'),
            'pretrain_smiles_file': config.dataset.get('pretrain_smiles_file', None),
        })

    mode_with_spectrum = stage

    log_dir = os.path.join(args.logdir, mode_with_spectrum)
    ckpt_dir = os.path.join(args.ckptdir, mode_with_spectrum)
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    ddp_barrier(distributed)

    if is_main:
        logger = get_logger('train', log_dir)
        writer = SummaryWriter(log_dir)
    else:
        logger = get_logger(f'train.rank{rank}', None)
        logger.setLevel(logging.WARNING)
        writer = NullWriter()

    logger.info(args)
    logger.info(config)
    logger.info(f'Training mode: {mode}')
    logger.info(f'stage: {stage}')
    if distributed:
        logger.info(f'DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}, device={args.device}')
    logger.info(f'Output directory: {mode_with_spectrum}')


    logger.info('Loading dataset...')
    if distributed and stage == 'graph2mol' and not is_main:
        processed_path, keys_path = expected_smiles_processed_paths(
            config.dataset, config.chem.atomic_numbers
        )
        logger.warning(f'[graph2mol] rank {rank} is waiting for rank 0 to complete LMDB preprocessing')
        wait_for_files([processed_path, keys_path], logger, f'[graph2mol] rank{rank} LMDB wait')
        dataset, subsets = get_dataset(config.dataset)
    else:
        dataset, subsets = get_dataset(config.dataset)


    atomic_numbers_config = config.chem.atomic_numbers
    if atomic_numbers_config == 'auto':

        if hasattr(dataset, 'detected_atomic_numbers') and dataset.detected_atomic_numbers:
            atomic_numbers = dataset.detected_atomic_numbers
            logger.info(f'Automatically detected atom types: {atomic_numbers}')
        else:

            atomic_numbers = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]
            logger.info(f'Using default atom types: {atomic_numbers}')
    else:
        atomic_numbers = list(atomic_numbers_config)
        logger.info(f'Using atom types specified by the configuration: {atomic_numbers}')


    config.chem.atomic_numbers = atomic_numbers


    dataset_name = config.dataset.get('name', 'msg')
    if dataset_name == 'msg_diffms':

        logger.info('msg_diffms contains encoded features; skipping graph transformation')
        featurizer = None
    elif dataset_name in ('msfile', 'smiles'):

        logger.info('Using FeaturizeMol2D')
        featurizer = FeaturizeMol2D(
            atomic_numbers=atomic_numbers,
            mol_bond_types=config.chem.mol_bond_types,
            use_mask_node=config.transform.get('use_mask_node', True),
            use_mask_edge=config.transform.get('use_mask_edge', False)
        )
    else:

        logger.info('Using FeaturizeMol')
        featurizer = FeaturizeMol(
            atomic_numbers=atomic_numbers,
            mol_bond_types=config.chem.mol_bond_types,
            use_mask_node=config.transform.get('use_mask_node', True),
            use_mask_edge=config.transform.get('use_mask_edge', False)
        )
    if featurizer is not None:
        transform = Compose([featurizer])
        dataset.transform = transform







    train_dataset = subsets['train']


    eval_split = config.train.get('eval_split', 'val')
    if eval_split == 'valid':
        logger.warning("Interpreting train.eval_split='valid' as 'val'; update the configuration to 'val'")
        eval_split = 'val'
    if eval_split not in ('val', 'test'):
        raise ValueError(f"train.eval_split must be 'val' or 'test'; received {eval_split!r}")
    val_dataset = subsets[eval_split]
    train_dataset = limit_dataset(train_dataset, args.max_train_samples, config.train.seed)
    val_dataset = limit_dataset(val_dataset, args.max_val_samples, config.train.seed + 1)


    #   graph2mol  ->  make_smiles_collate_with_cache(zmol_cache)
    #   ms2mol     ->  make_msg_diffms_collate_with_cache(zms_cache)

    if stage == 'graph2mol':
        from utils.dataset import ensure_cond_emb_cache, _cache_paths
        from utils.transforms import make_smiles_collate_with_cache

        align_ckpt_for_cache = (args.align_ckpt or './checkpoints/align/align.pt')
        if not os.path.exists(align_ckpt_for_cache):
            raise FileNotFoundError(
                f"graph2mol stage requires  align ckpt  to build  zmol cache.\n"
                f"expected path: {align_ckpt_for_cache}\n"
                f"finish training  align stage, the  best ckpt switching to nameas  align.pt place under  ./checkpoints/align/"
            )
        logger.info(f'[graph2mol] Building or loading with the alignment checkpoint:  zmol cache: {align_ckpt_for_cache}')

        cache_dir = getattr(config.dataset, 'cache_dir', './data/cache')
        zmol_cache_path = _cache_paths(cache_dir)['zmol']
        zmol_ready_path = zmol_cache_path + '.ready'

        if is_main and os.path.exists(zmol_cache_path) and os.path.getsize(zmol_cache_path) == 0:
            logger.warning(f'[graph2mol] found  0 bytes zmol cache, delete and rebuild: {zmol_cache_path}')
            os.remove(zmol_cache_path)
            if os.path.exists(zmol_ready_path):
                os.remove(zmol_ready_path)

        if distributed and (not is_main) and (not os.path.exists(zmol_ready_path)):
            logger.warning(f'[graph2mol] rank{rank} waiting for  rank0 building zmol cache: {zmol_cache_path}')
            wait_for_files([zmol_ready_path], logger, f'[graph2mol] rank{rank} zmol cache wait')
            try:
                zmol_cache = torch.load(zmol_cache_path, weights_only=False)
            except TypeError:
                zmol_cache = torch.load(zmol_cache_path)
            logger.warning(f'[graph2mol] rank{rank} loaded zmol cache: {len(zmol_cache)}  entries')
        elif os.path.exists(zmol_cache_path):
            logger.info(f'[graph2mol] zmol cache already exists  ->  load directly: {zmol_cache_path}')
            try:
                zmol_cache = torch.load(zmol_cache_path, weights_only=False)
            except TypeError:
                zmol_cache = torch.load(zmol_cache_path)
            logger.info(f'[graph2mol] loaded zmol cache: {len(zmol_cache)}  entries')
            if is_main and not os.path.exists(zmol_ready_path):
                with open(zmol_ready_path, 'w') as fp:
                    fp.write(str(time.time()))
        else:
            from tqdm import tqdm as _tqdm
            import pickle as _pk



            smiles_file = getattr(config.dataset, 'smiles_file', None)
            if smiles_file and os.path.exists(smiles_file):
                logger.info(f'[graph2mol] Zmol cache is unavailable; reading SMILES from {smiles_file}')
                import pandas as _pd
                df = _pd.read_csv(smiles_file)
                all_smiles = sorted(df['smiles'].dropna().unique().tolist())
                logger.info(f'[graph2mol] Building the Zmol cache from {len(all_smiles)} unique SMILES')
            else:
                logger.info('[graph2mol] Zmol cache and CSV are unavailable; scanning LMDB')
                all_smiles = set()
                total_idx = sum(len(sub.indices) if hasattr(sub, 'indices') else len(sub)
                                for sub in subsets.values())
                pbar = _tqdm(total=total_idx, desc='Extracting SMILES from LMDB', unit='mol', mininterval=2.0)
                for split_name, sub in subsets.items():
                    base_ds = sub.dataset if hasattr(sub, 'dataset') else sub
                    indices = sub.indices if hasattr(sub, 'indices') else range(len(base_ds))
                    if hasattr(base_ds, 'keys') and hasattr(base_ds, '_connect_db'):
                        if base_ds.db is None:
                            base_ds._connect_db()
                        with base_ds.db.begin() as txn:
                            for idx in indices:
                                raw = txn.get(base_ds.keys[idx])
                                pbar.update(1)
                                if raw is None:
                                    continue
                                d = _pk.loads(raw)
                                smi = getattr(d, 'smiles', None)
                                if smi:
                                    all_smiles.add(smi)
                    else:
                        for idx in indices:
                            d = base_ds[idx]
                            pbar.update(1)
                            smi = getattr(d, 'smiles', None)
                            if smi:
                                all_smiles.add(smi)
                pbar.close()
                all_smiles = sorted(all_smiles)
                logger.info(f'[graph2mol] {len(all_smiles)} unique SMILES  ->  building zmol cache')
            zmol_cache = ensure_cond_emb_cache(
                stage='graph2mol',
                align_ckpt_path=align_ckpt_for_cache,
                smiles_pool=all_smiles,
                cache_dir=cache_dir,
                device=args.device,
                batch_size=64,
            )
            if is_main:
                with open(zmol_ready_path, 'w') as fp:
                    fp.write(str(time.time()))
        collate_fn = make_smiles_collate_with_cache(zmol_cache)

    elif stage == 'ms2mol':

        from utils.dataset import ensure_cond_emb_cache, _cache_paths
        from utils.transforms import make_msg_diffms_collate_with_cache, make_smiles_collate_with_cache
        align_ckpt_for_cache = (args.align_ckpt or './checkpoints/align/align.pt')
        if not os.path.exists(align_ckpt_for_cache):
            raise FileNotFoundError(
                f"ms2mol stage requires  align ckpt  to build  zms cache.\n"
                f"expected path: {align_ckpt_for_cache}"
            )
        msg_root = config.dataset.get('root', './data/msg_diffms')
        cache_dir = getattr(config.dataset, 'cache_dir', './data/cache')
        logger.info(f'[ms2mol] Building or loading with the alignment checkpoint:  zms cache: {align_ckpt_for_cache}')
        zms_cache = ensure_cond_emb_cache(
            stage='ms2mol',
            align_ckpt_path=align_ckpt_for_cache,
            msg_root=msg_root,
            cache_dir=cache_dir,
            device=args.device,
            batch_size=64,
        )
        logger.info(f'[ms2mol] loaded zms cache: {len(zms_cache)}  entries')


        _need_zmol = False
        _ms_cfg = getattr(config.train, 'ms2mol', None) or {}
        _ms = _ms_cfg.get if isinstance(_ms_cfg, dict) else (lambda k, d=None: getattr(_ms_cfg, k, d))
        for sub in ('adapter', 'distill'):
            sub_cfg = _ms(sub, {}) or {}
            _sub = sub_cfg.get if isinstance(sub_cfg, dict) else (lambda k, d=None: getattr(sub_cfg, k, d))
            if bool(_sub('enabled', False)):
                _need_zmol = True
        zmol_target_cache = None
        if _need_zmol:
            zmol_cache_path = _cache_paths(cache_dir)['zmol']
            if not os.path.exists(zmol_cache_path):
                raise FileNotFoundError(
                    f"adapter/KD enabled but  zmol cache  does not exist: {zmol_cache_path}\n"
                    f"run  graph2mol one timesbuildingcomplete zmol cache"
                )


            try:
                zmol_target_cache = torch.load(zmol_cache_path, weights_only=False)
            except TypeError:
                zmol_target_cache = torch.load(zmol_cache_path)
            logger.info(f'[ms2mol] adapter/KD enabled  ->  loading zmol cache (canonical SMILES key): '
                        f'{len(zmol_target_cache)}  entries ({zmol_cache_path})')
        collate_fn = make_msg_diffms_collate_with_cache(zms_cache, zmol_target_cache=zmol_target_cache)

    elif stage == 'joint':


        from utils.dataset import ensure_cond_emb_cache, _cache_paths
        from utils.transforms import make_msg_diffms_collate_with_cache
        align_ckpt_for_cache = (args.align_ckpt or './checkpoints/align/align.pt')
        if not os.path.exists(align_ckpt_for_cache):
            raise FileNotFoundError(
                f"joint stage requires  align ckpt  to build  zmol/zms cache.\n"
                f"expected path: {align_ckpt_for_cache}"
            )
        msg_root = config.dataset.get('root', './data/msg_diffms')
        cache_dir = getattr(config.dataset, 'cache_dir', './data/cache')
        logger.info(f'[joint] Building or loading with the alignment checkpoint:  zms cache: {align_ckpt_for_cache}')
        zms_cache = ensure_cond_emb_cache(
            stage='ms2mol',
            align_ckpt_path=align_ckpt_for_cache,
            msg_root=msg_root,
            cache_dir=cache_dir,
            device=args.device,
            batch_size=64,
        )
        logger.info(f'[joint] loaded zms cache: {len(zms_cache)}  entries')



        zmol_target_cache_path = _cache_paths(cache_dir)['zmol']
        if not os.path.exists(zmol_target_cache_path):
            raise FileNotFoundError(
                f"joint stage requires  zmol cache as  MS batch  trainingtarget: {zmol_target_cache_path}\n"
                f"run  graph2mol stagebuildingthe  cache"
            )
        try:
            zmol_target_cache = torch.load(zmol_target_cache_path, weights_only=False)
        except TypeError:
            zmol_target_cache = torch.load(zmol_target_cache_path)
        logger.info(f'[joint] loaded zmol target cache: {len(zmol_target_cache)}  entries '
                    f'({zmol_target_cache_path})')
        collate_fn = make_msg_diffms_collate_with_cache(
            zms_cache, zmol_target_cache=zmol_target_cache
        )

    else:
        from utils.transforms import collate_with_spectrum_features
        collate_fn = collate_with_spectrum_features

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config.train.seed),
        drop_last=False,
    ) if distributed else None


    main_batch_size = int(config.train.batch_size)
    if stage == 'joint':
        joint_cfg_for_ms = cfg_get(config.train, 'joint', {}) or {}
        main_batch_size = int(cfg_get(
            joint_cfg_for_ms, 'batch_size_ms2', config.train.batch_size
        ))
    if main_batch_size <= 0:
        raise ValueError(f'maindata loader   batch_size must be positive, ; received  {main_batch_size}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=main_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        collate_fn=collate_fn
    )

    logger.info(f'trainingset: {len(train_dataset)}, evaluation set({eval_split}): {len(val_dataset)}, '
                f'main loader batch_size={main_batch_size}')


    val_subset_ratio = config.train.get('val_subset_ratio', 1.0)
    if val_subset_ratio < 1.0:
        val_subset_size = int(len(val_dataset) * val_subset_ratio)
        val_subset_generator = torch.Generator()
        val_subset_seed = int(config.train.get('val_subset_seed', config.train.seed))
        val_subset_generator.manual_seed(val_subset_seed)
        val_subset_indices = torch.randperm(
            len(val_dataset), generator=val_subset_generator
        )[:val_subset_size].tolist()
        val_subset = torch.utils.data.Subset(val_dataset, val_subset_indices)
        logger.info(f'Validation subset: {val_subset_size} ({val_subset_ratio*100:.1f}%), seed={val_subset_seed}')
    else:
        val_subset = val_dataset
        logger.info('Validation subset: using the full validation set')


    num_edge_types = len(config.chem.mol_bond_types) + 1


    logger.info('Creating model...')
    num_node_types = len(atomic_numbers) + 1


    model_result = create_model(config, num_node_types, num_edge_types, atomic_numbers)
    if isinstance(model_result, tuple):
        model, _ = model_result
    else:
        model = model_result

    model = model.to(args.device)

    # ============================================================

    # ============================================================
    use_adapter = False
    adapter_mse_w = 0.0
    use_kd = False
    teacher_model = None
    kd_weight = 0.0
    kd_temperature = 1.0
    teacher_condition_mode = 'default'
    if stage in ('ms2mol', 'joint'):
        ms2mol_cfg = getattr(config.train, 'ms2mol', None) or {}
        _ms = ms2mol_cfg.get if isinstance(ms2mol_cfg, dict) else (lambda k, d=None: getattr(ms2mol_cfg, k, d))
        adapter_cfg = _ms('adapter', {}) or {}
        _ad = adapter_cfg.get if isinstance(adapter_cfg, dict) else (lambda k, d=None: getattr(adapter_cfg, k, d))
        if bool(_ad('enabled', False)):
            use_adapter = True
            adapter_mse_w = float(_ad('mse_weight', 0.5))
            model.use_zms_adapter = True
            n_adapter = sum(p.numel() for p in model.zms_adapter.parameters())
            logger.info(f' [ms2mol/joint] Zms -> Zmol Adapter enabled')
            logger.info(f'  adapter parameter count: {n_adapter/1e6:.3f}M, MSE loss weights: {adapter_mse_w}')

        distill_cfg = _ms('distill', {}) or {}
        _dk = distill_cfg.get if isinstance(distill_cfg, dict) else (lambda k, d=None: getattr(distill_cfg, k, d))
        if bool(_dk('enabled', False)):
            use_kd = True
            kd_weight = float(_dk('weight', 1.0))
            kd_temperature = float(_dk('temperature', 2.0))
            teacher_condition_mode = str(
                cfg_get(config.train, 'teacher_condition_mode', 'default')
            ).lower()
            if teacher_condition_mode not in ('real', 'default'):
                raise ValueError(
                    'train.teacher_condition_mode must be real or default; '
                    f'received {teacher_condition_mode!r}'
                )
            kd_feature_weight = float(_dk('feature_weight', 0.0))
            kd_feature_layers = _dk('feature_layers', 'all')
            teacher_ckpt = _dk('teacher_ckpt', '') or ''

            if not teacher_ckpt:
                import glob, re
                g2m_dir = os.path.join(args.ckptdir, 'graph2mol')
                if os.path.isdir(g2m_dir):
                    cands = [(int(re.search(r'iter(\d+)\.pt$', f).group(1)), f)
                             for f in glob.glob(os.path.join(g2m_dir, 'graph2mol_iter*.pt'))
                             if re.search(r'iter(\d+)\.pt$', f)]
                    if cands:
                        cands.sort(reverse=True)
                        teacher_ckpt = cands[0][1]
            if not teacher_ckpt or not os.path.exists(teacher_ckpt):
                raise FileNotFoundError(f"KD enabled but cannot find teacher ckpt: {teacher_ckpt}")
            logger.info(f' [ms2mol/joint] Knowledge Distillation enabled')
            logger.info(f'  teacher_ckpt: {teacher_ckpt}, logits weight: {kd_weight}, T: {kd_temperature}')
            logger.info(f'  teacher_condition_mode: {teacher_condition_mode}')
            if kd_feature_weight > 0:
                logger.info(f'   Multi-layer feature distillation enabled (FitNet-style): feature_weight={kd_feature_weight}, '
                            f'layers={kd_feature_layers}')
                model.capture_layer_h_nodes = True
            else:
                model.capture_layer_h_nodes = False

            from models.model import FLASH as _FLASH
            import copy
            teacher_cfg = copy.deepcopy(config.model)
            teacher_model = _FLASH(teacher_cfg, num_node_types, num_edge_types, atomic_numbers).to(args.device)
            try:
                t_ck = torch.load(teacher_ckpt, map_location=args.device, weights_only=False)
            except TypeError:
                t_ck = torch.load(teacher_ckpt, map_location=args.device)
            t_state = t_ck['model'] if 'model' in t_ck else t_ck
            teacher_missing, teacher_unexpected = teacher_model.load_state_dict(
                t_state, strict=False
            )
            assert_release_checkpoint_keys(
                teacher_missing,
                teacher_unexpected,
                context='KD teacher',
                allow_graph_to_ms=True,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

            teacher_model.use_zms_adapter = False

            teacher_model.capture_layer_h_nodes = (kd_feature_weight > 0)
            logger.info(f'  teacher modelloadedand frozen')
        else:
            kd_feature_weight = 0.0
            kd_feature_layers = 'all'



    start_iter = 0
    resume_path = args.resume
    resume_checkpoint = None


    if args.auto_resume and resume_path is None:
        import glob as glob_mod
        ckpt_pattern = os.path.join(ckpt_dir, f'{mode_with_spectrum}_iter*.pt')
        ckpt_files = glob_mod.glob(ckpt_pattern)
        if ckpt_files:

            def extract_iter(path):
                basename = os.path.basename(path)

                try:
                    return int(basename.split('_iter')[-1].replace('.pt', ''))
                except ValueError:
                    return -1
            ckpt_files.sort(key=extract_iter)
            resume_path = ckpt_files[-1]
            logger.info(f'[auto_resume] foundlatestcheckpoint: {resume_path}')
        else:
            logger.info(f'[auto_resume] not foundcheckpoint, start training from iteration zero')

    if resume_path:
        logger.info(f'from checkpointresume: {resume_path}')

        try:
            resume_checkpoint = torch.load(resume_path, map_location=args.device, weights_only=False)
        except TypeError:

            resume_checkpoint = torch.load(resume_path, map_location=args.device)
        missing, unexpected = model.load_state_dict(resume_checkpoint['model'], strict=False)
        assert_release_checkpoint_keys(
            missing,
            unexpected,
            context=f'resume ({resume_path})',
            allow_graph_resume=(stage == 'graph2mol'),
        )
        if missing or unexpected:
            logger.info(f'  - missing keys while resuming training: {len(missing)}; unexpected keys: {len(unexpected)}')
        start_iter = resume_checkpoint.get('iteration', 0) + 1
    elif args.pretrained_ckpt:


        logger.info(f'from previous stage checkpoint loadingmodelweights: {args.pretrained_ckpt}')
        try:
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=args.device, weights_only=False)
        except TypeError:
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=args.device)
        pretrained_state = pretrained_ckpt['model'] if 'model' in pretrained_ckpt else pretrained_ckpt
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        assert_release_checkpoint_keys(
            missing,
            unexpected,
            context=f'pretrained ({args.pretrained_ckpt})',
            allow_graph_to_ms=(stage in ('ms2mol', 'joint')),
        )
        logger.info(f'  - missing keys (randomly initialized): {len(missing)}  '
                    + (f' examples: {missing[:3]}' if missing else ''))
        logger.info(f'  - unexpected keys (ignored): {len(unexpected)}  '
                    + (f' examples: {unexpected[:3]}' if unexpected else ''))
        if unexpected:
            logger.info('  The non-strict state-dict key set passed the release compatibility allowlist')
        start_iter = 0
    else:

        if stage in ('ms2mol', 'joint'):
            import glob, re
            g2m_dir = os.path.join(args.ckptdir, 'graph2mol')
            candidates = []
            if os.path.isdir(g2m_dir):
                for f in glob.glob(os.path.join(g2m_dir, 'graph2mol_iter*.pt')):
                    m = re.search(r'iter(\d+)\.pt$', f)
                    if m:
                        candidates.append((int(m.group(1)), f))
            if candidates:
                candidates.sort(reverse=True)
                latest_iter, latest_path = candidates[0]
                logger.info(f' {stage} stageautomaticallyloadinglatest graph2mol ckpt: {latest_path} (iter={latest_iter})')
                try:
                    pre_ckpt = torch.load(latest_path, map_location=args.device, weights_only=False)
                except TypeError:
                    pre_ckpt = torch.load(latest_path, map_location=args.device)
                pre_state = pre_ckpt['model'] if 'model' in pre_ckpt else pre_ckpt
                miss, unex = model.load_state_dict(pre_state, strict=False)
                assert_release_checkpoint_keys(
                    miss,
                    unex,
                    context=f'auto graph2mol pretrained ({latest_path})',
                    allow_graph_to_ms=True,
                )
                logger.info(f'  - missing keys: {len(miss)}; unexpected keys: {len(unex)}')
            else:
                logger.warning(f'  {stage} not found graph2mol ckpt(path {g2m_dir}), BFN backbonefrom zero initialization')
        start_iter = 0


    model = model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {num_params/1e6:.2f}M')


    optimizer_config = config.train.optimizer
    _optimizer_params = [p for p in model.parameters() if p.requires_grad]
    if not _optimizer_params:
        raise ValueError('No parameters with requires_grad=True were passed to the optimizer')
    if optimizer_config.type == 'adam':
        optimizer = torch.optim.Adam(
            _optimizer_params,
            lr=float(optimizer_config.lr),
            weight_decay=float(optimizer_config.weight_decay),
            betas=(float(optimizer_config.beta1), float(optimizer_config.beta2))
        )
    elif optimizer_config.type == 'adamw':
        optimizer = torch.optim.AdamW(
            _optimizer_params,
            lr=float(optimizer_config.lr),
            weight_decay=float(optimizer_config.weight_decay),
            betas=(float(optimizer_config.beta1), float(optimizer_config.beta2))
        )
    else:
        raise ValueError(f"unknown optimizer type: {optimizer_config.type}")


    scheduler_config = config.train.get('scheduler', None)
    scheduler = None
    if scheduler_config:
        if scheduler_config.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_config.factor,
                patience=scheduler_config.patience, min_lr=scheduler_config.min_lr
            )


    warmup_iters = config.train.get('warmup_iters', 1000)
    warmup_target_lrs = [float(pg['lr']) for pg in optimizer.param_groups]
    for pg, target_lr in zip(optimizer.param_groups, warmup_target_lrs):
        pg.setdefault('initial_lr', target_lr)
    target_lr_str = ', '.join(f'{lr:.3e}' for lr in warmup_target_lrs)
    logger.info(f'Warmup: before  {warmup_iters} stepslinearly warm up to  param_group_lrs=[{target_lr_str}]')

    if resume_checkpoint is not None:
        try:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            if scheduler and 'scheduler' in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint['scheduler'])
        except ValueError as exc:
            logger.warning(
                "The optimizer or scheduler state is incompatible with the current parameter groups; "
                f"restoring model weights only: {exc}"
            )



    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        logger.info('DDP initialized with find_unused_parameters=True')

    # ============================================================

    # ============================================================
    g2m_loader = None
    if stage == 'joint':
        from utils.dataset import get_dataset as _get_dataset
        from utils.dataset import ensure_cond_emb_cache, _cache_paths
        from utils.transforms import make_smiles_collate_with_cache
        from easydict import EasyDict as _ED
        logger.info('[joint] Building Graph2Mol data loader (SmilesDataset)...')
        joint_g2m_cfg = cfg_get(config.dataset, 'graph2mol', {}) or {}
        g2m_root = cfg_get(
            joint_g2m_cfg, 'root',
            cfg_get(
                config.dataset, 'graph2mol_root',
                cfg_get(config.dataset, 'pretrain_root', './data/pretrain'),
            ),
        )
        g2m_smiles_file = cfg_get(joint_g2m_cfg, 'smiles_file', None)
        if not g2m_smiles_file:
            g2m_smiles_file = cfg_get(config.dataset, 'graph2mol_smiles_file', None)
        if not g2m_smiles_file:
            g2m_smiles_file = cfg_get(config.dataset, 'pretrain_smiles_file', None)
        if not g2m_smiles_file:
            g2m_smiles_file = os.path.join(g2m_root, 'pretrain_smiles.csv')
        g2m_ds_cfg = _ED({
            'name': 'smiles',
            'root': g2m_root,
            'smiles_file': g2m_smiles_file,
            'max_atoms': cfg_get(joint_g2m_cfg, 'max_atoms', None),
            'split_seed': int(cfg_get(joint_g2m_cfg, 'split_seed', 2026)),
            'split_ratio': cfg_get(joint_g2m_cfg, 'split_ratio', [0.95, 0.025, 0.025]),
            'data_subset_ratio': float(cfg_get(joint_g2m_cfg, 'data_subset_ratio', 1.0)),
            'cache_dir': cfg_get(joint_g2m_cfg, 'cache_dir', config.dataset.cache_dir),
            'atomic_numbers': list(config.chem.atomic_numbers),
        })
        g2m_ds, g2m_subsets = _get_dataset(g2m_ds_cfg)
        from torch_geometric.transforms import Compose as PygCompose
        g2m_featurizer = FeaturizeMol2D(
            atomic_numbers=atomic_numbers,
            mol_bond_types=config.chem.mol_bond_types,
            use_mask_node=config.transform.get('use_mask_node', True),
            use_mask_edge=config.transform.get('use_mask_edge', False),
        )
        g2m_ds.transform = PygCompose([g2m_featurizer])

        align_for_zmol = args.align_ckpt or './checkpoints/align/align.pt'
        if not os.path.exists(align_for_zmol):
            raise FileNotFoundError(
                f"joint stage requires  align ckpt  to build  zmol cache: {align_for_zmol}"
            )
        cache_dir = cfg_get(g2m_ds_cfg, 'cache_dir', './data/cache')
        zmol_cache_path = _cache_paths(cache_dir)['zmol']
        if os.path.abspath(zmol_cache_path) == os.path.abspath(zmol_target_cache_path):


            zmol_cache = zmol_target_cache
            logger.info(f'[joint] reusing loaded  zmol target cache: {len(zmol_cache)}  entries '
                        f'({zmol_cache_path})')
        elif os.path.exists(zmol_cache_path):
            try:
                zmol_cache = torch.load(zmol_cache_path, weights_only=False)
            except TypeError:
                zmol_cache = torch.load(zmol_cache_path)
            logger.info(f'[joint] loaded zmol cache: {len(zmol_cache)}  entries')
        else:
            logger.info(f'[joint] zmol cache  does not exist  ->  from  smiles datasetbuilding(from  align ckpt: {align_for_zmol})')
            all_smiles = sorted({d.smiles for d in g2m_ds if hasattr(d, 'smiles') and d.smiles})
            logger.info(f'[joint] collected  {len(all_smiles)} unique SMILES, startingbuilding zmol cache ...')
            zmol_cache = ensure_cond_emb_cache(
                stage='graph2mol',
                align_ckpt_path=align_for_zmol,
                smiles_pool=all_smiles,
                cache_dir=cache_dir,
                device=args.device,
                batch_size=64,
            )
            logger.info(f'[joint] built zmol cache: {len(zmol_cache)}  entries')
        g2m_collate = make_smiles_collate_with_cache(zmol_cache)

        joint_cfg = getattr(config.train, 'joint', None) or {}
        joint_get = (joint_cfg.get if isinstance(joint_cfg, dict) else
                     lambda k, d=None: getattr(joint_cfg, k, d))
        g2m_bs = int(joint_get('batch_size_g2m', 64))
        g2m_sampler = DistributedSampler(
            g2m_subsets['train'],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config.train.seed) + 1009,
            drop_last=False,
        ) if distributed else None
        g2m_loader = DataLoader(
            g2m_subsets['train'], batch_size=g2m_bs,
            shuffle=(g2m_sampler is None), sampler=g2m_sampler,
            num_workers=config.train.num_workers, pin_memory=config.train.pin_memory,
            collate_fn=g2m_collate,
        )
        logger.info(f'[joint] graph2mol train loader: {len(g2m_subsets["train"])}  entries, batch_size={g2m_bs}')


    logger.info('Starting training...')
    train_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iterator = iter(train_loader)


    def _is_oom(e):
        return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()

    oom_count = 0

    for it in range(start_iter, config.train.max_iters):
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_epoch)
            train_iterator = iter(train_loader)
            batch = next(train_iterator)


        if batch is None:
            continue

        optimizer.zero_grad()


        if it < warmup_iters:
            warmup_scale = (it + 1) / warmup_iters
            for pg, target_lr in zip(optimizer.param_groups, warmup_target_lrs):
                pg['lr'] = target_lr * warmup_scale

        if it == start_iter:
            print_first_iter_debug(batch, config, logger)

        # ============================================================

        # ============================================================
        try:

            loss_dict = train_flash(model, batch, args.device, config, iteration=it, logger=logger)

            loss = loss_dict['loss']

            # ============================================================


            # ============================================================
            if use_adapter and stage in ('ms2mol', 'joint'):
                if hasattr(batch, 'zmol_target') and batch.zmol_target is not None:
                    adapter_out = unwrap_model(model)._last_adapter_out
                    zmol_target = batch.zmol_target.to(adapter_out.device)
                    mse = F.mse_loss(adapter_out, zmol_target)
                    loss = loss + adapter_mse_w * mse
                    loss_dict['loss_adapter_mse'] = mse.detach()
                    if it < 10:
                        logger.info(f"[Iter {it}] adapter MSE: {mse.item():.4f}, "
                                    f"weight={adapter_mse_w}")

            # ============================================================


            # ============================================================
            if use_kd and teacher_model is not None and stage in ('ms2mol', 'joint'):
                if hasattr(batch, 'zmol_target') and batch.zmol_target is not None:
                    with torch.no_grad():

                        _orig_use_adapter = teacher_model.use_zms_adapter
                        teacher_model.use_zms_adapter = False
                        try:


                            teacher_inst, teacher_ion = resolve_condition_indices(
                                batch, batch.zmol_target.size(0), args.device, config,
                                force_mode=teacher_condition_mode,
                            )
                            t_kwargs = dict(
                                node_types=batch.node_type.to(args.device),
                                edge_index=batch.halfedge_index.to(args.device),
                                batch_node=batch.node_type_batch.to(args.device),
                                batch_edge=batch.halfedge_type_batch.to(args.device),
                                instrument_type_idx=teacher_inst,
                                ionization_type_idx=teacher_ion,
                                cond_emb_cached=batch.zmol_target,
                            )

                            t_t = loss_dict.get('_t', None)
                            t_theta = loss_dict.get('_theta', None)
                            if t_t is not None and t_theta is not None:
                                t_kwargs['t'] = t_t
                                t_kwargs['edge_types_t'] = torch.zeros_like(batch.halfedge_type.to(args.device))
                                t_kwargs['edge_theta'] = t_theta
                                _ = teacher_model(**t_kwargs)
                                teacher_logits = getattr(teacher_model, '_last_x0_logits', None)
                            else:
                                teacher_logits = None
                        finally:
                            teacher_model.use_zms_adapter = _orig_use_adapter


                    student_logits = loss_dict.get('_edge_raw_logits', None)
                    if teacher_logits is not None and student_logits is not None:
                        # KL(student || teacher) with temperature
                        T = kd_temperature
                        s_log_softmax = F.log_softmax(student_logits / T, dim=-1)
                        t_softmax = F.softmax(teacher_logits / T, dim=-1)
                        kd_loss = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean') * (T * T)
                        loss = loss + kd_weight * kd_loss
                        loss_dict['loss_kd'] = kd_loss.detach()
                        if it < 10:
                            logger.info(f"[Iter {it}] KD: {kd_loss.item():.4f}, "
                                        f"weight={kd_weight}, T={T}")

                    # ============================================================


                    # ============================================================
                    if kd_feature_weight > 0:
                        s_layers = getattr(unwrap_model(model), '_last_layer_h_nodes', None)
                        t_layers = getattr(teacher_model, '_last_layer_h_nodes', None)
                        if s_layers is not None and t_layers is not None \
                                and len(s_layers) == len(t_layers):
                            n_layers = len(s_layers)

                            if kd_feature_layers == 'all':
                                layer_idx = list(range(n_layers))
                            elif kd_feature_layers == 'last':
                                layer_idx = [n_layers - 1]
                            elif isinstance(kd_feature_layers, (list, tuple)):
                                layer_idx = [i for i in kd_feature_layers if 0 <= i < n_layers]
                            else:
                                layer_idx = list(range(n_layers))

                            feat_losses = []
                            for i in layer_idx:
                                feat_losses.append(F.mse_loss(s_layers[i], t_layers[i].detach()))
                            feat_loss = sum(feat_losses) / max(1, len(feat_losses))
                            loss = loss + kd_feature_weight * feat_loss
                            loss_dict['loss_feat_kd'] = feat_loss.detach()
                            if it < 10:
                                logger.info(f"[Iter {it}] feature KD ({len(layer_idx)} layers): "
                                            f"{feat_loss.item():.4f}, weight={kd_feature_weight}")

            loss_dict['loss'] = loss


            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[Iter {it}] Detected  NaN/Inf loss, skipping this update(this may indicate numerical overflow)")
                optimizer.zero_grad()
                continue

            # ============================================================


            #   2. graph2mol forward + backward

            # ============================================================
            if stage == 'joint' and g2m_loader is not None:
                joint_cfg = getattr(config.train, 'joint', None) or {}
                joint_get = (joint_cfg.get if isinstance(joint_cfg, dict) else
                             lambda k, d=None: getattr(joint_cfg, k, d))
                w_ms2 = float(joint_get('weight_ms2mol', 5.0))


                (w_ms2 * loss).backward()
                loss_dict['loss_ms2'] = loss.detach()


                try:
                    g2m_batch = next(g2m_iterator)
                except (NameError, StopIteration):
                    g2m_iterator = iter(g2m_loader)
                    g2m_batch = next(g2m_iterator)
                if g2m_batch is not None:
                    _orig_stage = config.model.stage
                    model_core_joint = unwrap_model(model)
                    _orig_use_adapter = getattr(model_core_joint, 'use_zms_adapter', False)
                    try:
                        config.model.stage = 'graph2mol'

                        model_core_joint.use_zms_adapter = False
                        ld_g2m = train_flash(
                            model, g2m_batch, args.device, config,
                            iteration=None, logger=None,
                        )
                    finally:
                        model_core_joint.use_zms_adapter = _orig_use_adapter
                        config.model.stage = _orig_stage
                    loss_g2m = ld_g2m['loss']
                    if not (torch.isnan(loss_g2m) or torch.isinf(loss_g2m)):
                        loss_g2m.backward()
                        loss_dict['loss_g2m'] = loss_g2m.detach()

                        loss_dict['loss'] = (loss_g2m.detach() + w_ms2 * loss.detach())
                        if it < 10:
                            logger.info(f"[Iter {it}] joint: loss_g2m={loss_g2m.item():.4f}, "
                                        f"loss_ms2={loss.item():.4f}, w_ms2={w_ms2}, "
                                        f"total={loss_dict['loss'].item():.4f}")
                    else:
                        logger.warning(f"[Iter {it}] g2m loss NaN/Inf, skipped g2m backward")
            else:
                loss.backward()
        except RuntimeError as _e:
            if _is_oom(_e):
                oom_count += 1
                logger.warning(f"[Iter {it}] CUDA OOM (cumulative {oom_count} times), skipping this step")
                optimizer.zero_grad(set_to_none=True)
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise



        model_core = unwrap_model(model)
        if hasattr(model_core, 'spectrum_projector') and model_core.spectrum_projector is not None:
            spec_grads = [p.grad.norm().item() for n, p in model_core.spectrum_projector.named_parameters() if p.grad is not None]
            if spec_grads:
                writer.add_scalar('grad/spectrum_projector_mean', np.mean(spec_grads), it)
                writer.add_scalar('grad/spectrum_projector_max', np.max(spec_grads), it)


        if hasattr(model_core, 'peaks_encoder') and model_core.peaks_encoder is not None:
            peaks_grads = [p.grad.norm().item() for n, p in model_core.peaks_encoder.named_parameters() if p.grad is not None]
            if peaks_grads:
                writer.add_scalar('grad/peaks_encoder_mean', np.mean(peaks_grads), it)
                writer.add_scalar('grad/peaks_encoder_max', np.max(peaks_grads), it)


        if hasattr(model_core, 'node_embedder'):
            node_emb_grad = model_core.node_embedder.weight.grad
            if node_emb_grad is not None:
                writer.add_scalar('grad/node_embedder', node_emb_grad.norm().item(), it)


        all_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        if all_grads:
            writer.add_scalar('grad/total_mean', np.mean(all_grads), it)
            writer.add_scalar('grad/total_max', np.max(all_grads), it)


        if it < 10:
            logger.info(f"[Iter {it}] Gradient analysis:")


            if hasattr(model_core, 'spectrum_projector') and model_core.spectrum_projector is not None:
                spec_grads = [p.grad.norm().item() for n, p in model_core.spectrum_projector.named_parameters() if p.grad is not None]
                if spec_grads:
                    logger.info(f"  [mass spectrumencoder-dreams] gradient norm: mean={np.mean(spec_grads):.6f}, maximum={np.max(spec_grads):.6f}")


            if hasattr(model_core, 'peaks_encoder') and model_core.peaks_encoder is not None:
                peaks_grads = [p.grad.norm().item() for n, p in model_core.peaks_encoder.named_parameters() if p.grad is not None]
                if peaks_grads:
                    logger.info(f"  [mass spectrumencoder-origin] gradient norm: mean={np.mean(peaks_grads):.6f}, maximum={np.max(peaks_grads):.6f}")

            if hasattr(model_core, 'condition_embedding'):
                for name, param in model_core.condition_embedding.named_parameters():
                    if param.grad is not None:
                        logger.info(f"  [condition embedding] {name}: gradient norm={param.grad.norm().item():.6f}")

            if hasattr(model, 'node_embedder') and node_emb_grad is not None:
                logger.info(f"  [node embedding/Molecular formula] gradient norm={node_emb_grad.norm().item():.6f}")

            if hasattr(model_core, 'edge_predictor') and model_core.edge_predictor is not None:
                edge_grads = [p.grad.norm().item() for n, p in model_core.edge_predictor.named_parameters() if p.grad is not None]
                if edge_grads:
                    logger.info(f"  [edge-prediction head] gradient norm: mean={np.mean(edge_grads):.6f}")

        if config.train.get('max_grad_norm'):
            clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()


        if '_edge_logits' in loss_dict and '_edge_types_true' in loss_dict:
            with torch.no_grad():
                edge_logits = loss_dict['_edge_logits']
                edge_types_true = loss_dict['_edge_types_true']
                pred = edge_logits.argmax(dim=-1)


                pred_dist = torch.bincount(pred, minlength=5).tolist()
                true_dist = torch.bincount(edge_types_true, minlength=5).tolist()

                total_acc = (pred == edge_types_true).float().mean().item()
                bond_mask = edge_types_true > 0
                bond_acc = ((pred == edge_types_true) & bond_mask).sum().item() / bond_mask.sum().item() if bond_mask.sum() > 0 else 0


                batch_edge = loss_dict.get('_batch_edge', None)
                if batch_edge is not None:
                    num_mols = batch_edge.max().item() + 1
                    mol_exact_match = 0
                    for mol_idx in range(num_mols):
                        mol_mask = batch_edge == mol_idx
                        if mol_mask.sum() > 0:
                            mol_pred = pred[mol_mask]
                            mol_true = edge_types_true[mol_mask]
                            if (mol_pred == mol_true).all():
                                mol_exact_match += 1
                    mol_acc = mol_exact_match / num_mols if num_mols > 0 else 0
                else:
                    mol_acc = 0


                logger.info(f'  true distribution: {true_dist} | predicted distribution: {pred_dist} | TotalAcc={total_acc:.4f} BondAcc={bond_acc:.4f} MolAcc={mol_acc:.4f}')


                writer.add_scalar('train/total_acc', total_acc, it)
                writer.add_scalar('train/bond_acc', bond_acc, it)
                writer.add_scalar('train/mol_acc', mol_acc, it)


                if config.train.get('log_edge_distribution', True):

                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(5)
                    width = 0.35
                    edge_type_names = ['NoBond(0)', 'Single(1)', 'Double(2)', 'Triple(3)', 'Aromatic(4)']


                    total_true = sum(true_dist)
                    total_pred = sum(pred_dist)
                    true_ratio = [t/total_true if total_true > 0 else 0 for t in true_dist]
                    pred_ratio = [p/total_pred if total_pred > 0 else 0 for p in pred_dist]

                    bars1 = ax.bar(x - width/2, true_ratio, width, label='Ground Truth', color='steelblue', alpha=0.8)
                    bars2 = ax.bar(x + width/2, pred_ratio, width, label='Prediction', color='coral', alpha=0.8)

                    ax.set_xlabel('Edge Type')
                    ax.set_ylabel('Ratio')
                    ax.set_title(f'Iter {it}: TotalAcc={total_acc:.4f} BondAcc={bond_acc:.4f} MolAcc={mol_acc:.4f}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(edge_type_names)
                    ax.legend()
                    ax.set_ylim(0, 1.0)


                    for bar, val in zip(bars1, true_dist):
                        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='steelblue')
                    for bar, val in zip(bars2, pred_dist):
                        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='coral')

                    plt.tight_layout()
                    writer.add_figure('distribution/edge_type_comparison', fig, it)
                    plt.close(fig)


            del loss_dict['_edge_logits']
            del loss_dict['_edge_types_true']
            if '_batch_edge' in loss_dict:
                del loss_dict['_batch_edge']
        # ==========================================

        loss_str = ' '.join([f'{key}={value.item() if torch.is_tensor(value) else value:.4f}' for key, value in loss_dict.items() if not key.startswith('_')])
        logger.info(f'[{it}/{config.train.max_iters}] {loss_str}')
        for key, value in loss_dict.items():
            if not key.startswith('_'):
                if torch.is_tensor(value):
                    writer.add_scalar(f'train/{key}', value.item(), it)
                else:
                    writer.add_scalar(f'train/{key}', value, it)


        if it % config.train.val_freq == 0 and it > 0:
            val_loss_for_scheduler = None
            eval_model = unwrap_model(model)
            eval_model.eval()

            if distributed:
                val_local_indices = list(range(rank, len(val_subset), world_size))
                val_eval_dataset = torch.utils.data.Subset(val_subset, val_local_indices)
            else:
                val_eval_dataset = val_subset

            val_loader_subset = DataLoader(
                val_eval_dataset,
                batch_size=main_batch_size,
                shuffle=False,
                num_workers=config.train.num_workers,
                pin_memory=config.train.pin_memory,
                collate_fn=collate_fn
            )

            if is_main:
                logger.info(f'[{it}] Starting validation...')
                logger.info(
                    f'[{it}] distributionformulavalidation: Total samples={len(val_subset)}, '
                    f'world_size={world_size if distributed else 1}, rank0samplethis ={len(val_eval_dataset)}'
                )

            val_seed = int(config.train.get('seed', 0)) + int(it) * 1009 + rank * 1000003
            torch.manual_seed(val_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(val_seed)

            val_metric_keys = ['bfn_loss', 'total', 'loss']
            val_sums = {key: 0.0 for key in val_metric_keys}
            val_count = 0
            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader_subset,
                    desc=f'validation iter {it}' if not distributed else f'validation iter {it} rank{rank}',
                    leave=False,
                    disable=distributed and not is_main,
                )
                for val_batch in val_pbar:
                    if val_batch is None:
                        continue
                    val_loss_dict = train_flash(eval_model, val_batch, args.device, config)
                    for key in val_metric_keys:
                        if key in val_loss_dict:
                            value = val_loss_dict[key]
                            val_sums[key] += float(value.item() if torch.is_tensor(value) else value)
                    val_count += 1
                    if 'loss' in val_loss_dict:
                        val_pbar.set_postfix({'loss': val_loss_dict['loss'].item()})

            val_reduce = torch.tensor(
                [val_sums['bfn_loss'], val_sums['total'], val_sums['loss'], float(val_count)],
                dtype=torch.float64,
                device=args.device,
            )
            if distributed:
                dist.all_reduce(val_reduce, op=dist.ReduceOp.SUM)

            global_val_count = int(val_reduce[3].item())
            if global_val_count == 0:
                if is_main:
                    logger.warning(f'[{it}] all validation-set  batch were all filtered, skipping this validation pass/reconstruction evaluation')
            else:
                avg_val_losses = {
                    'bfn_loss': val_reduce[0].item() / global_val_count,
                    'total': val_reduce[1].item() / global_val_count,
                    'loss': val_reduce[2].item() / global_val_count,
                }
                val_loss_for_scheduler = avg_val_losses.get('loss', None)

                if is_main:
                    val_loss_str = ' '.join([f'{key}={value:.4f}' for key, value in avg_val_losses.items()])
                    logger.info(f'[{it}] validation loss: {val_loss_str}')

                    for key, value in avg_val_losses.items():
                        writer.add_scalar(f'val/{key}', value, it)


                    eval_num_samples = config.train.get('eval_num_samples', 100)
                    logger.info(f'[{it}] Starting reconstruction evaluation(per  samplethis generate{eval_num_samples} results)...')
                    recon_metrics = evaluate_reconstruction(
                        model=eval_model,
                        val_dataset=val_subset,
                        device=args.device,
                        config=config,
                        mode=mode,
                        atomic_numbers=atomic_numbers,
                        logger=logger,
                        collate_fn=collate_fn,
                        num_samples=eval_num_samples,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                        eval_iteration=it,
                    )
                else:
                    eval_num_samples = config.train.get('eval_num_samples', 100)
                    recon_metrics = evaluate_reconstruction(
                        model=eval_model,
                        val_dataset=val_subset,
                        device=args.device,
                        config=config,
                        mode=mode,
                        atomic_numbers=atomic_numbers,
                        logger=logger,
                        collate_fn=collate_fn,
                        num_samples=eval_num_samples,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                        eval_iteration=it,
                    )

                if is_main:

                    writer.add_scalar('val/recon_success_rate', recon_metrics.get('recon_success_rate', 0.0), it)
                    writer.add_scalar('val/edge_accuracy', recon_metrics.get('edge_accuracy', 0.0), it)
                    writer.add_scalar('val/bond_accuracy', recon_metrics.get('bond_accuracy', 0.0), it)
                    writer.add_scalar('val/mol_accuracy', recon_metrics.get('mol_accuracy', 0.0), it)

                    if 'pairwise_top1' in recon_metrics:
                        writer.add_scalar('val/pairwise_top1', recon_metrics['pairwise_top1'], it)
                    if 'cos_sim_mean' in recon_metrics:
                        writer.add_scalar('val/cos_sim_mean', recon_metrics['cos_sim_mean'], it)

                    if 'top1_mol_accuracy' in recon_metrics:
                        writer.add_scalar('val/top1_mol_accuracy', recon_metrics['top1_mol_accuracy'], it)
                    if 'top10_mol_accuracy' in recon_metrics:
                        writer.add_scalar('val/top10_mol_accuracy', recon_metrics['top10_mol_accuracy'], it)
                    # ==========================================================

            if distributed:
                ddp_barrier(distributed)

            if scheduler and val_loss_for_scheduler is not None:
                scheduler.step(val_loss_for_scheduler)


        if it % config.train.save_freq == 0 and it > 0:
            if is_main:
                ckpt_path = os.path.join(ckpt_dir, f'{mode_with_spectrum}_iter{it}.pt')
                torch.save({
                    'config': config,
                    'model': unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'iteration': it,
                    'mode': mode,
                    'stage': stage,
                }, ckpt_path)
                logger.info(f'Saving checkpoint: {ckpt_path}')
            ddp_barrier(distributed)

    logger.info('Training complete!')
    writer.close()
    if distributed:
        ddp_barrier(distributed)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
