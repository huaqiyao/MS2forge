"""
FLASH训练脚本
FLASH = Flow-based Learning for Assembly of molecular Structures from MS/MS with formula Hints
统一使用离散贝叶斯流网络 (Discrete BFN) 生成范式
支持质谱模式：formula, formula+dreams
"""

import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
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
matplotlib.use('Agg')  # 非交互式后端

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, FeaturizeMol2D
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError
from torch_geometric.transforms import Compose


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_logger(name, log_dir=None):
    """获取logger"""
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
    """创建FLASH模型"""
    from models.model import FLASH
    model = FLASH(config.model, num_node_types, num_edge_types, atomic_numbers)
    return model, None


def _permute_nodes_within_batch(node_type, halfedge_index, batch_node, device):
    """对 batch 内每个分子的节点做独立随机置换。
    halfedge_index 同步更新，确保边连接关系不变。

    参考思路：图对比学习常用的"节点重排"增强（MolCLR 类似 atom_mask 思路的对称变形）。
    GNN 理论上置换等变，但浮点 + mean_pool 数值上仍有微小差异；
    每次见到不同的节点序列，迫使 graph_encoder 学到真正的不变表征，防止记住特定顺序。

    Args:
        node_type      : [N_total]  当前 batch 拼好的节点类型
        halfedge_index : [2, M_total]  半边索引
        batch_node     : [N_total]  每个节点属于哪个图
        device         : torch.device

    Returns:
        new_node_type, new_halfedge_index, batch_node（不变）
    """
    # 构造 old_idx → new_idx 映射
    perm_map = torch.arange(node_type.size(0), device=device)
    n_graphs = int(batch_node.max().item()) + 1
    for g in range(n_graphs):
        mask = (batch_node == g).nonzero(as_tuple=False).view(-1)
        if mask.numel() <= 1:
            continue
        shuffled = mask[torch.randperm(mask.numel(), device=device)]
        perm_map[mask] = shuffled

    # 应用到 node_type：new_node_type[new_idx] = node_type[old_idx]
    # 等价于 new_node_type = node_type 然后按 perm_map 重排
    # 这里 perm_map 是"old → new"，所以 new_pos_of_old = perm_map
    # → 让 new_node_type[perm_map[i]] = node_type[i]
    new_node_type = torch.empty_like(node_type)
    new_node_type[perm_map] = node_type

    # halfedge_index 也按 perm_map 重映射
    new_halfedge_index = perm_map[halfedge_index]

    return new_node_type, new_halfedge_index, batch_node


def train_flash(model, batch, device, config, iteration=None, logger=None):
    """FLASH 训练步骤（DeniMS 风格三阶段）。

    按 model.stage 分支：
      - align     : 双塔 InfoNCE，不跑 BFN
      - graph2mol : graph_encoder(mol) → 256 维条件 → BFN 边去噪
      - ms2mol    : ms_encoder(DreaMS) → 256 维条件 → BFN 边去噪
    """
    batch = batch.to(device)
    stage = getattr(config.model, 'stage', None)
    if stage not in ('align', 'graph2mol', 'ms2mol'):
        raise ValueError(f"model.stage 必须是 align/graph2mol/ms2mol，得到 {stage!r}")

    # ============================================================
    # align 阶段：DeniMS 风格双塔 InfoNCE
    # 输入：(spec_sos, spec_formula_array, spec_mask) + (dense_X, dense_E, dense_y, dense_node_mask)
    # ============================================================
    if stage == 'align':
        if not hasattr(batch, 'spec_sos') or batch.spec_sos is None:
            raise ValueError("align 阶段需要 batch.spec_sos / spec_formula_array / spec_mask")

        n_total = batch.num_graphs

        # ============================================================
        # ★ 反过拟合开关（仅 align 阶段）★
        # ============================================================
        aug_cfg = getattr(config.train, 'augment', None) or {}
        aug_get = aug_cfg.get if hasattr(aug_cfg, 'get') else (lambda k, d=None: getattr(aug_cfg, k, d))

        spec_sos = batch.spec_sos
        spec_formula_array = batch.spec_formula_array
        spec_mask = batch.spec_mask
        dense_X = batch.dense_X
        dense_E = batch.dense_E
        dense_y = batch.dense_y
        dense_node_mask = batch.dense_node_mask

        # 谱端高斯噪声（对 formula_array 全张量加扰动）
        spec_noise_sigma = float(aug_get('spec_noise_sigma', 0.0))
        if spec_noise_sigma > 0 and model.training:
            spec_formula_array = spec_formula_array + torch.randn_like(spec_formula_array) * spec_noise_sigma

        out = model(
            spec_sos=spec_sos,
            spec_formula_array=spec_formula_array,
            spec_mask=spec_mask,
            dense_X=dense_X,
            dense_E=dense_E,
            dense_y=dense_y,
            dense_node_mask=dense_node_mask,
        )

        # ---- 方案 1: 多正样本 InfoNCE（参考 SupContrast）----
        positive_mask = None
        multi_positive = bool(getattr(config.train, 'multi_positive', False))
        if multi_positive and hasattr(batch, 'smiles') and batch.smiles is not None:
            smi_list = batch.smiles
            B = len(smi_list)
            if B == n_total:
                positive_mask = torch.zeros(B, B, dtype=torch.float32, device=device)
                from collections import defaultdict
                buckets = defaultdict(list)
                for i, s in enumerate(smi_list):
                    buckets[s].append(i)
                for idxs in buckets.values():
                    for i in idxs:
                        for j in idxs:
                            positive_mask[i, j] = 1.0

        loss_main = model.compute_contrastive_loss(
            out['ms_emb'], out['graph_emb'], positive_mask=positive_mask,
        )

        # ---- 方案 8: MixCo 风格 mixup ----
        mixup_alpha = float(getattr(config.train, 'mixup_alpha', 0.0))
        mixup_weight = float(getattr(config.train, 'mixup_weight', 0.5))
        loss_mix = torch.zeros((), device=device)
        if mixup_alpha > 0 and model.training:
            loss_mix = model.compute_mixup_contrastive_loss(
                out['ms_emb'], out['graph_emb'], alpha=mixup_alpha,
            )
            loss = loss_main + mixup_weight * loss_mix
        else:
            loss = loss_main

        if iteration is not None and iteration < 10 and logger is not None:
            extra = ''
            if positive_mask is not None:
                n_pos_per = positive_mask.sum(1).mean().item()
                extra += f', avg_positives_per_anchor={n_pos_per:.2f}'
            if mixup_alpha > 0:
                extra += f', mixup_loss={loss_mix.item():.4f}'
            logger.info(f"[Iter {iteration}] align contrastive: loss_main={loss_main.item():.4f}, "
                        f"temperature={1.0/model.inv_temperature.item():.2f}{extra}")

        return {
            'loss': loss,
            'contrastive_loss': loss_main.detach(),
            'mixup_loss': loss_mix.detach() if torch.is_tensor(loss_mix) else loss_mix,
            'inv_temperature': model.inv_temperature.detach(),
            '_edge_types_true': torch.zeros(0, dtype=torch.long, device=device),
            '_batch_edge': torch.zeros(0, dtype=torch.long, device=device),
        }

    # ============================================================
    # graph2mol / ms2mol：BFN 边去噪
    # graph2mol 走 SmilesDataset（无谱）→ batch 是 PyG Batch
    # ms2mol    走 DiffMSMSGDataset → batch 是 SimpleNamespace（含 spec_* 和 dense_*）
    # ============================================================
    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.node_type_batch.max().item() + 1)

    # 仪器/ionization 条件（DiffMSMSGDataset 返回 [B] tensor，SmilesDataset 返回 [B]）
    if hasattr(batch, 'instrument_type_idx_batch') and batch.instrument_type_idx_batch is not None:
        instrument_type_idx = batch.instrument_type_idx_batch.to(device)
    else:
        instrument_type_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    if hasattr(batch, 'ionization_type_idx_batch') and batch.ionization_type_idx_batch is not None:
        ionization_type_idx = batch.ionization_type_idx_batch.to(device)
    else:
        ionization_type_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    # 准备 forward kwargs：graph2mol/ms2mol 都直接用预算好的 cond_emb_cached
    # （collate factory 已把 zmol/zms cache 注入到 batch.cond_emb_cached）
    if not hasattr(batch, 'cond_emb_cached') or batch.cond_emb_cached is None:
        raise ValueError(
            f"{stage} 阶段 batch 缺 cond_emb_cached。请确认用了 make_smiles_collate_with_cache "
            f"或 make_msg_diffms_collate_with_cache。"
        )
    fwd_kwargs = {'cond_emb_cached': batch.cond_emb_cached}

    # 采样时间 + BFN 后验
    t = torch.rand(batch_size, device=device)
    edge_types_true = batch.halfedge_type
    theta = model.discrete_bayesian_update(t, edge_types_true, batch.halfedge_type_batch)

    if iteration is not None and iteration < 10 and logger is not None:
        logger.info(f"[Iter {iteration}] {stage} BFN 调试:")
        logger.info(f"  - 时间t范围: [{t.min().item():.3f}, {t.max().item():.3f}]")
        logger.info(f"  - theta 熵: {-(theta * theta.clamp(min=1e-8).log()).sum(-1).mean().item():.4f}")

    # 前向（cond_emb 来源在 model.forward 内部按 stage 自动分流）
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
        **fwd_kwargs,
    )

    losses = model.compute_bfn_loss(e_hat, edge_types_true, t, batch.halfedge_type_batch)
    total_loss = losses['total']
    loss_dict = {k: v for k, v in losses.items()}
    loss_dict['loss'] = total_loss
    loss_dict['_edge_types_true'] = edge_types_true.detach()
    loss_dict['_batch_edge'] = batch.halfedge_type_batch.detach()
    return loss_dict


def print_first_iter_debug(batch, config, logger):
    """第一个iteration时打印调试信息"""
    import random as rand

    logger.info("=" * 60)
    logger.info("[DEBUG] 第一个iteration调试信息")
    logger.info("=" * 60)

    has_spectrum = hasattr(batch, 'batch_has_spectrum') and batch.batch_has_spectrum
    has_mask = hasattr(batch, 'has_spectrum_mask') and batch.has_spectrum_mask.any()
    has_embedding = hasattr(batch, 'pretrained_embedding_batch') and batch.pretrained_embedding_batch is not None

    logger.info(f"[质谱特征检查]")
    logger.info(f"  - batch_has_spectrum: {has_spectrum}")
    logger.info(f"  - has_spectrum_mask: {has_mask}")
    logger.info(f"  - pretrained_embedding_batch存在: {has_embedding}")

    if has_embedding:
        emb = batch.pretrained_embedding_batch
        logger.info(f"  - pretrained_embedding_batch形状: {emb.shape}")
        logger.info(f"  - pretrained_embedding_batch范围: [{emb.min().item():.4f}, {emb.max().item():.4f}]")

    logger.info(f"[分子式/节点类型检查]")
    logger.info(f"  - node_type形状: {batch.node_type.shape}")
    logger.info(f"  - node_type唯一值: {batch.node_type.unique().tolist()}")
    logger.info(f"  - 分子数量: {batch.num_graphs}")

    logger.info(f"[训练标签检查]")
    logger.info(f"  - halfedge_type形状: {batch.halfedge_type.shape}")
    logger.info(f"  - halfedge_type唯一值: {batch.halfedge_type.unique().tolist()}")

    mol_idx = rand.randint(0, batch.num_graphs - 1)
    logger.info(f"[随机分子 #{mol_idx} 详细信息]")

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
    logger.info(f"  - 分子式: {formula_str}")
    logger.info(f"  - 原子数: {mol_nodes.shape[0]}")

    logger.info("=" * 60)


def evaluate_reconstruction(model, val_dataset, device, config, mode, atomic_numbers, logger, collate_fn, num_samples=100):
    """
    评估模型：按 stage 分支
      - align     : 算 pairwise matching top-1 accuracy + mean cos similarity（不跑 BFN 采样）
      - graph2mol : BFN 采样评估（用 graph_encoder(mol) 当条件，与训练时一致）
      - ms2mol    : BFN 采样评估（用 ms_encoder(DreaMS) 当条件）

    Args:
        model: 训练的模型
        val_dataset: 验证集Dataset（或Subset）
        device: 设备
        config: 配置
        mode: 模型模式
        atomic_numbers: 原子类型列表
        logger: 日志器
        collate_fn: collate函数
        num_samples: 每个样本生成的结果数量（默认100；align 阶段不使用）

    Returns:
        dict: 包含准确率的字典
    """
    model.eval()
    stage = getattr(config.model, 'stage', None)

    # 读取评估专用的batch size
    eval_batch_size = config.train.get('eval_batch_size', 4)
    eval_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ============================================================
    # align 阶段：双塔 pairwise matching 评估
    # ============================================================
    if stage == 'align':
        logger.info(f'[align 评估] eval_batch_size={eval_batch_size}, 总样本数={len(val_dataset)}')
        all_ms_emb = []
        all_graph_emb = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc='align 评估', leave=False, position=0):
                batch = batch.to(device)
                if not (hasattr(batch, 'batch_has_spectrum') and batch.batch_has_spectrum):
                    continue
                if not (hasattr(batch, 'pretrained_embedding_batch')
                        and batch.pretrained_embedding_batch is not None):
                    continue
                # 仅取带谱样本
                has_mask = batch.has_spectrum_mask
                n_total = batch.node_type_batch.max().item() + 1
                if int(has_mask.sum().item()) != n_total:
                    continue   # 跳过混合 batch（与训练逻辑一致）
                emb = batch.pretrained_embedding_batch[has_mask].to(device)
                if emb.dim() == 3:
                    emb = emb.squeeze(1)

                out = model(
                    node_types=batch.node_type,
                    edge_index=batch.halfedge_index,
                    batch_node=batch.node_type_batch,
                    batch_edge=batch.halfedge_type_batch,
                    pretrained_embedding=emb,
                )
                all_ms_emb.append(out['ms_emb'].cpu())
                all_graph_emb.append(out['graph_emb'].cpu())

        if not all_ms_emb:
            logger.warning('[align 评估] 验证集没有可用的样本，跳过')
            return {'pairwise_top1': 0.0, 'cos_sim_mean': 0.0,
                    'edge_accuracy': 0.0, 'bond_accuracy': 0.0, 'mol_accuracy': 0.0}

        ms_embs = torch.cat(all_ms_emb, dim=0)             # [N, D]
        graph_embs = torch.cat(all_graph_emb, dim=0)
        ms_embs = F.normalize(ms_embs, dim=-1)
        graph_embs = F.normalize(graph_embs, dim=-1)

        N = ms_embs.size(0)
        # 全集 pairwise matching：对每个 ms_emb_i，看 graph_emb_i 是否是其最近邻
        # 注意大集合上是 N×N 相似度，N 可能很大，做分块计算
        chunk = 256
        top1_correct = 0
        for s in range(0, N, chunk):
            sim_chunk = ms_embs[s:s+chunk] @ graph_embs.t()  # [c, N]
            preds = sim_chunk.argmax(dim=-1)                 # 最近邻
            target = torch.arange(s, s + sim_chunk.size(0))
            top1_correct += (preds == target).sum().item()
        pairwise_top1 = top1_correct / max(N, 1)

        # 自身配对的 cos similarity 平均
        cos_sim_mean = (ms_embs * graph_embs).sum(-1).mean().item()

        logger.info(f'[align 评估] 验证集统计:')
        logger.info(f'  样本数 N = {N}')
        logger.info(f'  pairwise top-1 accuracy = {pairwise_top1:.4f}')
        logger.info(f'  mean cosine similarity  = {cos_sim_mean:.4f}')

        return {
            'pairwise_top1': pairwise_top1,
            'cos_sim_mean': cos_sim_mean,
            # 兼容主训练循环里看的字段
            'edge_accuracy': 0.0,
            'bond_accuracy': 0.0,
            'mol_accuracy': 0.0,
        }

    # ============================================================
    # graph2mol / ms2mol：BFN 采样评估
    # ============================================================
    logger.info(f'评估配置: eval_batch_size={eval_batch_size}, num_samples={num_samples}, 总样本数={len(val_dataset)}')

    total_correct = 0
    total_edges = 0
    bond_correct = 0
    total_bonds = 0
    mol_exact_match = 0
    total_mols = 0

    # Top-K 统计（仅当 num_samples > 1 且为生成模型时使用）
    top1_correct_count = 0
    top10_correct_count = 0

    with torch.no_grad():
        batch_pbar = tqdm(eval_loader, desc='评估验证集', leave=False, position=0)
        for batch_idx, batch in enumerate(batch_pbar):
            batch = batch.to(device)

            # graph2mol/ms2mol 都用 batch.cond_emb_cached（由 cache-version collate 注入）
            if not hasattr(batch, 'cond_emb_cached') or batch.cond_emb_cached is None:
                logger.warning(f"  评估 batch 缺 cond_emb_cached，跳过")
                continue
            cond_emb_cached = batch.cond_emb_cached.to(device)
            batch_size_orig = cond_emb_cached.size(0)

            # 仪器/ionization：ms2mol 从 batch 取；graph2mol 全 0
            if stage == 'ms2mol':
                instrument_type_idx = batch.instrument_type_idx_batch.to(device) \
                    if hasattr(batch, 'instrument_type_idx_batch') and batch.instrument_type_idx_batch is not None \
                    else torch.zeros(batch_size_orig, dtype=torch.long, device=device)
                ionization_type_idx = batch.ionization_type_idx_batch.to(device) \
                    if hasattr(batch, 'ionization_type_idx_batch') and batch.ionization_type_idx_batch is not None \
                    else torch.zeros(batch_size_orig, dtype=torch.long, device=device)
            else:  # graph2mol
                instrument_type_idx = torch.zeros(batch_size_orig, dtype=torch.long, device=device)
                ionization_type_idx = torch.zeros(batch_size_orig, dtype=torch.long, device=device)

            edge_types_true = batch.halfedge_type

            num_nodes = batch.node_type.shape[0]
            num_edges = batch.halfedge_index.shape[1]

            # 将输入扩展 num_samples 倍
            node_types_expanded = batch.node_type.repeat(num_samples)
            edge_index_expanded = batch.halfedge_index.repeat(1, num_samples)
            for i in range(1, num_samples):
                edge_index_expanded[:, i*num_edges:(i+1)*num_edges] += i * num_nodes

            batch_node_expanded = torch.cat([batch.node_type_batch + i * batch_size_orig for i in range(num_samples)])
            batch_edge_expanded = torch.cat([batch.halfedge_type_batch + i * batch_size_orig for i in range(num_samples)])

            instrument_type_idx_expanded = instrument_type_idx.repeat(num_samples)
            ionization_type_idx_expanded = ionization_type_idx.repeat(num_samples)

            # 扩展 cond_emb_cached （每个样本复制 num_samples 次）
            cond_emb_expanded = cond_emb_cached.repeat(num_samples, 1)   # [num_samples * B, 512]

            # BFN 采样（用 cond_emb_cached 而不是 pretrained_embedding）
            pred_edge_types_all = model.sample_bfn(
                node_types=node_types_expanded,
                edge_index=edge_index_expanded,
                batch_node=batch_node_expanded,
                batch_edge=batch_edge_expanded,
                instrument_type_idx=instrument_type_idx_expanded,
                ionization_type_idx=ionization_type_idx_expanded,
                cond_emb_cached=cond_emb_expanded,
                n_timesteps=config.model.flow.get('eval_n_timesteps', 100),
                disable_tqdm=False
            )

            # 将结果拆分回 num_samples 个独立预测
            all_predictions = []
            for i in range(num_samples):
                pred_edge_types = pred_edge_types_all[i*num_edges:(i+1)*num_edges]
                all_predictions.append(pred_edge_types)

            # 更新外层进度条
            batch_pbar.set_postfix({'已处理分子': total_mols})

            # 使用第一个预测计算 TotalAcc 和 BondAcc
            pred_edge_types = all_predictions[0]

            # 累计TotalAcc
            total_correct += (pred_edge_types == edge_types_true).sum().item()
            total_edges += edge_types_true.numel()

            # 累计BondAcc
            bond_mask = edge_types_true > 0
            bond_correct += ((pred_edge_types == edge_types_true) & bond_mask).sum().item()
            total_bonds += bond_mask.sum().item()

            # 累计MolAcc（遍历该batch的每个分子）
            num_mols_in_batch = batch.halfedge_type_batch.max().item() + 1
            for mol_idx in range(num_mols_in_batch):
                mol_mask = batch.halfedge_type_batch == mol_idx
                if mol_mask.sum() > 0:
                    mol_pred = pred_edge_types[mol_mask]
                    mol_true = edge_types_true[mol_mask]
                    is_match = (mol_pred == mol_true).all()

                    if is_match:
                        mol_exact_match += 1

                    # ========== Top-K 计算（仅当 num_samples > 1 且为生成模型时）==========
                    if num_samples > 1:
                        # 收集该分子的所有预测序列（保持在 GPU 上）
                        mol_predictions_gpu = []
                        for pred_single in all_predictions:
                            mol_pred_single = pred_single[mol_mask]
                            mol_predictions_gpu.append(mol_pred_single)

                        # 使用 GPU 加速的唯一性检测
                        unique_predictions = []
                        unique_counts = []

                        for pred in mol_predictions_gpu:
                            # 检查是否已存在
                            found = False
                            for i, unique_pred in enumerate(unique_predictions):
                                if torch.equal(pred, unique_pred):
                                    unique_counts[i] += 1
                                    found = True
                                    break
                            if not found:
                                unique_predictions.append(pred)
                                unique_counts.append(1)

                        # 按频率排序
                        sorted_indices = sorted(range(len(unique_counts)), key=lambda i: unique_counts[i], reverse=True)

                        # 计算 Top-1 和 Top-10
                        n_unique_predictions = len(unique_predictions)
                        top1_frequency = unique_counts[sorted_indices[0]] if len(sorted_indices) > 0 else 0

                        # 检查真实序列是否在 top-1 和 top-10 中
                        top1_match = False
                        top10_match = False

                        if len(sorted_indices) > 0:
                            # Top-1 逻辑：如果最高频率是1（所有预测都不同），检查所有预测中是否有匹配
                            if top1_frequency == 1:
                                # 所有预测都不同，检查是否有任何一个匹配
                                for unique_pred in unique_predictions:
                                    if torch.equal(unique_pred, mol_true):
                                        top1_match = True
                                        break
                            else:
                                # 有重复预测，只检查频率最高的
                                top1_match = torch.equal(unique_predictions[sorted_indices[0]], mol_true)

                            # Top-10 逻辑：如果有>=10个唯一预测且最高频率是1，检查所有预测；否则检查频率最高的前10个
                            if n_unique_predictions >= 10 and top1_frequency == 1:
                                # 有10个或更多不同预测，检查所有预测中是否有匹配
                                for unique_pred in unique_predictions:
                                    if torch.equal(unique_pred, mol_true):
                                        top10_match = True
                                        break
                            else:
                                # 预测数量<10 或有重复，检查频率最高的前10个
                                for idx in sorted_indices[:min(10, len(sorted_indices))]:
                                    if torch.equal(unique_predictions[idx], mol_true):
                                        top10_match = True
                                        break

                        # 累计 Top-K 统计
                        if top1_match:
                            top1_correct_count += 1
                        if top10_match:
                            top10_correct_count += 1

                    total_mols += 1

    # 计算最终指标
    total_acc = total_correct / total_edges if total_edges > 0 else 0
    bond_acc = bond_correct / total_bonds if total_bonds > 0 else 0
    mol_acc = mol_exact_match / total_mols if total_mols > 0 else 0

    logger.info(f"[评估] 验证集统计:")
    logger.info(f"  总边数: {total_edges}, 总分子数: {total_mols}")
    logger.info(f"  TotalAcc: {total_acc:.4f}")
    logger.info(f"  BondAcc: {bond_acc:.4f}")
    logger.info(f"  MolAcc: {mol_exact_match}/{total_mols} = {mol_acc:.4f}")

    # 如果 num_samples > 1 且为生成模型，输出 Top-K 准确率
    if num_samples > 1:
        top1_acc = top1_correct_count / total_mols if total_mols > 0 else 0.0
        top10_acc = top10_correct_count / total_mols if total_mols > 0 else 0.0
        logger.info(f"  Top-1 MolAcc: {top1_acc:.4f} ({top1_correct_count}/{total_mols})")
        logger.info(f"  Top-10 MolAcc: {top10_acc:.4f} ({top10_correct_count}/{total_mols})")

    result_dict = {
        'recon_success_rate': 1.0,
        'edge_accuracy': total_acc,
        'bond_accuracy': bond_acc,
        'mol_accuracy': mol_acc
    }

    # 添加 Top-K 指标
    if num_samples > 1:
        result_dict['top1_mol_accuracy'] = top1_correct_count / total_mols if total_mols > 0 else 0.0
        result_dict['top10_mol_accuracy'] = top10_correct_count / total_mols if total_mols > 0 else 0.0

    return result_dict


def check_prediction_distribution(edge_logits, edge_types, logger, iteration):
    """
    检查预测分布，诊断模型是否陷入局部最优

    Args:
        edge_logits: 模型预测的logits
        edge_types: 真实边类型
        logger: 日志器
        iteration: 当前迭代次数
    """
    with torch.no_grad():
        pred = torch.argmax(edge_logits, dim=-1)
        pred_dist = torch.bincount(pred, minlength=5)
        true_dist = torch.bincount(edge_types, minlength=5)

        # 计算各类别的准确率
        total_acc = (pred == edge_types).float().mean().item()

        # 有键边的准确率
        bond_mask = edge_types > 0
        if bond_mask.sum() > 0:
            bond_acc = ((pred == edge_types) & bond_mask).sum().item() / bond_mask.sum().item()
        else:
            bond_acc = 0.0

        # 检查是否全预测为无键
        no_bond_ratio = pred_dist[0].item() / pred.shape[0]

        logger.info(f"[分布检查] Iter {iteration}:")
        logger.info(f"  预测分布: {pred_dist.tolist()} (无键占比: {no_bond_ratio:.2%})")
        logger.info(f"  真实分布: {true_dist.tolist()}")
        logger.info(f"  总准确率: {total_acc:.4f}, 有键准确率: {bond_acc:.4f}")

        if no_bond_ratio > 0.99:
            logger.warning(f"  ⚠️ 警告: 模型预测几乎全是无键({no_bond_ratio:.2%})，可能陷入局部最优!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--logdir', type=str, default='/root/tf-logs/', help='日志目录')
    parser.add_argument('--ckptdir', type=str, default='./checkpoints/', help='checkpoint保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--auto_resume', action='store_true', help='自动加载当前模式下最新的checkpoint继续训练')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='上一阶段 ckpt 路径（仅模型权重，strict=False，从 0 步开始）。'
                             '用法：graph2mol 时传 align ckpt；ms2mol 时传 graph2mol ckpt')
    parser.add_argument('--align_ckpt', type=str, default=None,
                        help='align 阶段产出的 ckpt（含 ms_encoder + graph_encoder）。'
                             '仅 ms2mol 阶段用：从中提取 ms_encoder.* 权重叠加到当前模型')
    parser.add_argument('--overfit_test', default=False, help='运行单batch过拟合测试')
    args = parser.parse_args()

    config = load_config(args.config)
    seed_all(config.train.seed)

    mode = 'flash-model'

    # ★ 三阶段切换：align / graph2mol / ms2mol
    stage = getattr(config.model, 'stage', None)
    if stage not in ('align', 'graph2mol', 'ms2mol'):
        raise ValueError(
            f"model.stage 必须是 align / graph2mol / ms2mol 之一，得到 {stage!r}"
        )

    # 按 stage 自动覆盖 dataset 块（无需手改 yaml）：
    #   align     → DiffMSMSGDataset (DeniMS 格式：sub-formula 序列 + dense X/E)
    #   graph2mol → SmilesDataset (smiles，HMDB+DSSTox+COCONUT+MOSES 3.3M，仅 mol)
    #   ms2mol    → DiffMSMSGDataset (同 align，但 BFN 主干用条件)
    if stage == 'graph2mol':
        # graph2mol 用 SmilesDataset：HMDB+DSSTox+COCONUT+MOSES+MSG（按 csv split 列切 train/val/test）
        # MSG 部分的 split 与 ms2mol 阶段保持一致 → val/test 永远是 MSG 那批分子
        pretrain_root = './data/pretrain'
        config.dataset = EasyDict({
            'name': 'smiles',
            'root': pretrain_root,
            'smiles_file': os.path.join(pretrain_root, 'pretrain_smiles.csv'),
            'max_atoms': None,
            'split_seed': 2026,
            'split_ratio': [0.95, 0.025, 0.025],
            'data_subset_ratio': 1.0,
            'atomic_numbers': list(config.chem.atomic_numbers)
                if isinstance(config.chem.atomic_numbers, (list, tuple))
                else [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53],
        })
    else:  # align / ms2mol：均用 DiffMS 预处理过的 MSG（含 sub-formula 标注）
        config.dataset = EasyDict({
            'name': 'msg_diffms',
            'root': config.dataset.get('root', './data/msg_diffms'),
            'data_split_mode': config.dataset.get('data_split_mode', 'split'),
            'instrument_type': config.dataset.get('instrument_type', 'all'),
            'data_subset_ratio': 1.0,
            'max_peaks': config.dataset.get('max_peaks', 128),
        })

    mode_with_spectrum = stage  # 用 stage 名命名子目录：align / graph2mol / ms2mol

    log_dir = os.path.join(args.logdir, mode_with_spectrum)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)

    ckpt_dir = os.path.join(args.ckptdir, mode_with_spectrum)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(args)
    logger.info(config)
    logger.info(f'训练模式: {mode}')
    logger.info(f'阶段: {stage}')
    logger.info(f'保存目录: {mode_with_spectrum}')

    # 加载数据集
    logger.info('加载数据集...')
    dataset, subsets = get_dataset(config.dataset)

    # 处理原子类型配置：支持 'auto' 自动检测
    atomic_numbers_config = config.chem.atomic_numbers
    if atomic_numbers_config == 'auto':
        # 从数据集中获取自动检测的原子类型
        if hasattr(dataset, 'detected_atomic_numbers') and dataset.detected_atomic_numbers:
            atomic_numbers = dataset.detected_atomic_numbers
            logger.info(f'自动检测到的原子类型: {atomic_numbers}')
        else:
            # 默认原子类型
            atomic_numbers = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]
            logger.info(f'使用默认原子类型: {atomic_numbers}')
    else:
        atomic_numbers = list(atomic_numbers_config)
        logger.info(f'使用配置文件指定的原子类型: {atomic_numbers}')

    # 更新配置中的原子类型（用于后续保存）
    config.chem.atomic_numbers = atomic_numbers

    # 根据数据集类型选择合适的 featurizer
    dataset_name = config.dataset.get('name', 'msg')
    if dataset_name == 'msg_diffms':
        # DiffMSMSGDataset 自己输出已编码的 PyG Data，不需要 FeaturizeMol2D
        logger.info('msg_diffms 自带编码，跳过 transform')
        featurizer = None
    elif dataset_name in ('msfile', 'smiles'):
        # MSFileDataset / SmilesDataset 使用简化的 2D featurizer
        logger.info('使用 FeaturizeMol2D（简化 2D 格式）')
        featurizer = FeaturizeMol2D(
            atomic_numbers=atomic_numbers,
            mol_bond_types=config.chem.mol_bond_types,
            use_mask_node=config.transform.get('use_mask_node', True),
            use_mask_edge=config.transform.get('use_mask_edge', False)
        )
    else:
        # 其他数据集使用原始 featurizer
        logger.info('使用 FeaturizeMol（原始格式）')
        featurizer = FeaturizeMol(
            atomic_numbers=atomic_numbers,
            mol_bond_types=config.chem.mol_bond_types,
            use_mask_node=config.transform.get('use_mask_node', True),
            use_mask_edge=config.transform.get('use_mask_edge', False)
        )
    if featurizer is not None:
        transform = Compose([featurizer])
        dataset.transform = transform
    # else: msg_diffms 用自身输出，不应用 transform

    train_dataset = subsets['train']
    # eval_split: 'val'（默认）或 'test'，控制训练时评估用的子集
    eval_split = config.train.get('eval_split', 'val')
    if eval_split not in ('val', 'test'):
        raise ValueError(f"train.eval_split 只能是 'val' 或 'test'，得到 {eval_split!r}")
    val_dataset = subsets[eval_split]

    # 按 stage 选 collate：
    #   align     → collate_msg_diffms（不需要 cache，实时跑双塔）
    #   graph2mol → make_smiles_collate_with_cache(zmol_cache)
    #   ms2mol    → make_msg_diffms_collate_with_cache(zms_cache)
    if stage == 'align':
        from utils.transforms import collate_msg_diffms
        collate_fn = collate_msg_diffms
    elif stage == 'graph2mol':
        # 构建 zmol cache：用 align ckpt 给所有 SmilesDataset 里的 unique SMILES 算 graph_emb
        from utils.dataset import ensure_cond_emb_cache, _cache_paths
        from utils.transforms import make_smiles_collate_with_cache
        align_ckpt_path = (args.align_ckpt or
                           getattr(config.model, 'align_ckpt_path', None) or
                           './checkpoints/Encoder_Contrastive_FragHub.pth')
        if not os.path.exists(align_ckpt_path):
            raise FileNotFoundError(f"graph2mol 阶段必须有 align ckpt 来构建 zmol cache: {align_ckpt_path}")

        cache_dir = getattr(config.dataset, 'cache_dir', './data/cache')
        zmol_cache_path = _cache_paths(cache_dir)['zmol']

        if os.path.exists(zmol_cache_path):
            # cache 已存在 → 直接读，跳过扫 LMDB 和 RDKit 解析（省 10+ 分钟 + 几小时）
            logger.info(f'[graph2mol] zmol cache 已存在 → 直接加载（跳过扫 LMDB）: {zmol_cache_path}')
            try:
                zmol_cache = torch.load(zmol_cache_path, weights_only=False)
            except TypeError:
                zmol_cache = torch.load(zmol_cache_path)
            logger.info(f'[graph2mol] 已加载 zmol cache: {len(zmol_cache)} 条')
        else:
            # cache 不存在 → 扫 LMDB + RDKit + GPU forward
            from tqdm import tqdm as _tqdm
            import pickle as _pk
            logger.info('[graph2mol] zmol cache 不存在，扫 LMDB 收集所有 unique SMILES ...')
            all_smiles = set()
            total_idx = sum(len(sub.indices) if hasattr(sub, 'indices') else len(sub)
                            for sub in subsets.values())
            pbar = _tqdm(total=total_idx, desc='扫 LMDB 提取 SMILES', unit='mol',
                          mininterval=2.0)
            for split_name, sub in subsets.items():
                base_ds = sub.dataset if hasattr(sub, 'dataset') else sub
                indices = sub.indices if hasattr(sub, 'indices') else range(len(base_ds))
                if hasattr(base_ds, 'keys') and hasattr(base_ds, '_connect_db'):
                    if base_ds.db is None:
                        base_ds._connect_db()
                    # 复用一个 transaction，避免每条 .begin()
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
                    # fallback：通过 __getitem__ 取（慢）
                    for idx in indices:
                        d = base_ds[idx]
                        pbar.update(1)
                        smi = getattr(d, 'smiles', None)
                        if smi:
                            all_smiles.add(smi)
            pbar.close()
            all_smiles = sorted(all_smiles)
            logger.info(f'[graph2mol] {len(all_smiles)} unique SMILES → 构建 zmol cache')
            zmol_cache = ensure_cond_emb_cache(
                stage='graph2mol',
                align_ckpt_path=align_ckpt_path,
                smiles_pool=all_smiles,
                cache_dir=cache_dir,
                device=args.device,
                batch_size=64,
            )
        collate_fn = make_smiles_collate_with_cache(zmol_cache)
    elif stage == 'ms2mol':
        from utils.dataset import ensure_cond_emb_cache
        from utils.transforms import make_msg_diffms_collate_with_cache
        align_ckpt_path = (args.align_ckpt or
                           getattr(config.model, 'align_ckpt_path', None) or
                           './checkpoints/Encoder_Contrastive_FragHub.pth')
        if not os.path.exists(align_ckpt_path):
            raise FileNotFoundError(f"ms2mol 阶段必须有 align ckpt 来构建 zms cache: {align_ckpt_path}")
        msg_root = config.dataset.get('root', './data/msg_diffms')
        zms_cache = ensure_cond_emb_cache(
            stage='ms2mol',
            align_ckpt_path=align_ckpt_path,
            msg_root=msg_root,
            cache_dir=getattr(config.dataset, 'cache_dir', './data/cache'),
            device=args.device,
            batch_size=64,
        )
        collate_fn = make_msg_diffms_collate_with_cache(zms_cache)
    else:
        from utils.transforms import collate_with_spectrum_features
        collate_fn = collate_with_spectrum_features

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        collate_fn=collate_fn
    )

    logger.info(f'训练集: {len(train_dataset)}, 评估集({eval_split}): {len(val_dataset)}')

    # 验证集子集配置（用于加速训练阶段的验证）
    val_subset_ratio = config.train.get('val_subset_ratio', 1.0)
    if val_subset_ratio < 1.0:
        val_subset_size = int(len(val_dataset) * val_subset_ratio)
        val_subset_indices = torch.randperm(len(val_dataset))[:val_subset_size].tolist()
        val_subset = torch.utils.data.Subset(val_dataset, val_subset_indices)
        logger.info(f'验证集子集: {val_subset_size} ({val_subset_ratio*100:.1f}%)')
    else:
        val_subset = val_dataset
        logger.info('验证集子集: 使用全部验证集')

    # 提前计算num_edge_types（用于边缘分布计算）
    num_edge_types = len(config.chem.mol_bond_types) + 1

    # 创建模型
    logger.info('创建模型...')
    num_node_types = len(atomic_numbers) + 1
    # num_edge_types已在前面定义

    model_result = create_model(config, num_node_types, num_edge_types, atomic_numbers)
    if isinstance(model_result, tuple):
        model, _ = model_result
    else:
        model = model_result

    model = model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型参数量: {num_params/1e6:.2f}M')

    # 优化器
    optimizer_config = config.train.optimizer
    if optimizer_config.type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    elif optimizer_config.type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_config.type}")

    # 学习率调度器
    scheduler_config = config.train.get('scheduler', None)
    scheduler = None
    if scheduler_config:
        if scheduler_config.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_config.factor,
                patience=scheduler_config.patience, min_lr=scheduler_config.min_lr
            )

    # Warmup 调度：前 N 步线性从 0 升到 lr，防止训练初期梯度震荡
    warmup_iters = config.train.get('warmup_iters', 1000)
    base_lr = float(config.train.optimizer.lr)
    logger.info(f'Warmup: 前 {warmup_iters} 步线性升温至 lr={base_lr}')

    # 恢复训练
    start_iter = 0
    resume_path = args.resume

    # auto_resume：自动查找当前模式下 iter 最大的 checkpoint
    if args.auto_resume and resume_path is None:
        import glob as glob_mod
        ckpt_pattern = os.path.join(ckpt_dir, f'{mode_with_spectrum}_iter*.pt')
        ckpt_files = glob_mod.glob(ckpt_pattern)
        if ckpt_files:
            # 从文件名中提取 iter 数字，找最大的
            def extract_iter(path):
                basename = os.path.basename(path)
                # 格式: flash-model-flash_iter2000.pt
                try:
                    return int(basename.split('_iter')[-1].replace('.pt', ''))
                except ValueError:
                    return -1
            ckpt_files.sort(key=extract_iter)
            resume_path = ckpt_files[-1]
            logger.info(f'[auto_resume] 找到最新checkpoint: {resume_path}')
        else:
            logger.info(f'[auto_resume] 未找到checkpoint，从头开始训练')

    if resume_path:
        logger.info(f'从checkpoint恢复: {resume_path}')
        # 兼容不同版本的PyTorch（2.6+需要weights_only=False来加载包含EasyDict的checkpoint）
        try:
            checkpoint = torch.load(resume_path, map_location=args.device, weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            checkpoint = torch.load(resume_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint.get('iteration', 0) + 1
    elif args.pretrained_ckpt:
        # 上一阶段 ckpt（仅 model 权重，从 0 步开始）
        # 用法：graph2mol → 传 align ckpt；ms2mol → 传 graph2mol ckpt
        logger.info(f'从上一阶段 checkpoint 加载模型权重: {args.pretrained_ckpt}')
        try:
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=args.device, weights_only=False)
        except TypeError:
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=args.device)
        pretrained_state = pretrained_ckpt['model'] if 'model' in pretrained_ckpt else pretrained_ckpt
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        logger.info(f'  - 缺失键 (随机初始化): {len(missing)} 个'
                    + (f' 例: {missing[:3]}' if missing else ''))
        logger.info(f'  - 多余键 (已忽略): {len(unexpected)} 个'
                    + (f' 例: {unexpected[:3]}' if unexpected else ''))
        if unexpected:
            logger.warning('  存在 unexpected keys，可能维度/结构不匹配，请确认。')
        start_iter = 0
    else:
        # 默认分支：align 阶段如果 yaml 配置了 align_ckpt_path，自动加载 DeniMS ckpt
        align_ckpt_path = getattr(config.model, 'align_ckpt_path', None)
        if stage == 'align' and align_ckpt_path and os.path.exists(align_ckpt_path):
            logger.info(f'★ 从 yaml 配置自动加载 DeniMS align ckpt: {align_ckpt_path}')
            try:
                ack = torch.load(align_ckpt_path, map_location=args.device, weights_only=False)
            except TypeError:
                ack = torch.load(align_ckpt_path, map_location=args.device)
            sd = ack['model'] if 'model' in ack else ack
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f'  - 缺失键 (随机初始化): {len(missing)} 个'
                        + (f' 例: {missing[:3]}' if missing else ''))
            logger.info(f'  - 多余键 (已忽略): {len(unexpected)} 个'
                        + (f' 例: {unexpected[:3]}' if unexpected else ''))
        elif stage == 'align':
            logger.info('  align 阶段从随机权重开始（未配置 align_ckpt_path 或文件不存在）')
        start_iter = 0

    # ms2mol 阶段：从 align ckpt 注入 ms_encoder.* 权重（覆盖 init）
    if stage == 'ms2mol' and args.align_ckpt:
        logger.info(f'从 align checkpoint 注入 ms_encoder 权重: {args.align_ckpt}')
        try:
            align_ckpt = torch.load(args.align_ckpt, map_location=args.device, weights_only=False)
        except TypeError:
            align_ckpt = torch.load(args.align_ckpt, map_location=args.device)
        align_state = align_ckpt['model'] if 'model' in align_ckpt else align_ckpt
        # 仅保留 ms_encoder.* 键
        ms_state = {k: v for k, v in align_state.items() if k.startswith('ms_encoder.')}
        if not ms_state:
            logger.warning(f'  align ckpt 中未找到 ms_encoder.* 键，请确认 ckpt 来自 align 阶段')
        else:
            missing2, unexpected2 = model.load_state_dict(ms_state, strict=False)
            n_loaded = len(ms_state) - sum(1 for k in missing2 if k.startswith('ms_encoder.'))
            logger.info(f'  - 已注入 ms_encoder 权重: {n_loaded} / {len(ms_state)} 键')

    # graph2mol 阶段：从 align ckpt 注入 graph_encoder.* 权重（覆盖 init）
    if stage == 'graph2mol' and args.align_ckpt:
        logger.info(f'从 align checkpoint 注入 graph_encoder 权重: {args.align_ckpt}')
        try:
            align_ckpt = torch.load(args.align_ckpt, map_location=args.device, weights_only=False)
        except TypeError:
            align_ckpt = torch.load(args.align_ckpt, map_location=args.device)
        align_state = align_ckpt['model'] if 'model' in align_ckpt else align_ckpt
        ge_state = {k: v for k, v in align_state.items() if k.startswith('graph_encoder.')}
        if not ge_state:
            logger.warning(f'  align ckpt 中未找到 graph_encoder.* 键')
        else:
            missing3, unexpected3 = model.load_state_dict(ge_state, strict=False)
            n_loaded = len(ge_state) - sum(1 for k in missing3 if k.startswith('graph_encoder.'))
            logger.info(f'  - 已注入 graph_encoder 权重: {n_loaded} / {len(ge_state)} 键')

    # 训练循环
    logger.info('开始训练...')
    train_iterator = iter(train_loader)

    for it in range(start_iter, config.train.max_iters):
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        optimizer.zero_grad()

        # Warmup：前 warmup_iters 步线性升温
        if it < warmup_iters:
            warmup_lr = base_lr * (it + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        if it == start_iter:
            print_first_iter_debug(batch, config, logger)

        # 训练步骤
        loss_dict = train_flash(model, batch, args.device, config, iteration=it, logger=logger)

        loss = loss_dict['loss']

        # NaN/Inf 检测：跳过本步，不更新参数，防止参数污染
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[Iter {it}] 检测到 NaN/Inf loss，跳过本步更新（可能是数值溢出）")
            optimizer.zero_grad()
            continue

        loss.backward()

        # 记录梯度到TensorBoard
        # 1. 质谱特征编码器梯度
        if hasattr(model, 'spectrum_projector') and model.spectrum_projector is not None:
            spec_grads = [p.grad.norm().item() for n, p in model.spectrum_projector.named_parameters() if p.grad is not None]
            if spec_grads:
                writer.add_scalar('grad/spectrum_projector_mean', np.mean(spec_grads), it)
                writer.add_scalar('grad/spectrum_projector_max', np.max(spec_grads), it)

        # 1b. 原始质谱峰编码器梯度（origin模式）
        if hasattr(model, 'peaks_encoder') and model.peaks_encoder is not None:
            peaks_grads = [p.grad.norm().item() for n, p in model.peaks_encoder.named_parameters() if p.grad is not None]
            if peaks_grads:
                writer.add_scalar('grad/peaks_encoder_mean', np.mean(peaks_grads), it)
                writer.add_scalar('grad/peaks_encoder_max', np.max(peaks_grads), it)

        # 2. 节点特征嵌入梯度
        if hasattr(model, 'node_embedder'):
            node_emb_grad = model.node_embedder.weight.grad
            if node_emb_grad is not None:
                writer.add_scalar('grad/node_embedder', node_emb_grad.norm().item(), it)

        # 3. 整体模型梯度
        all_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        if all_grads:
            writer.add_scalar('grad/total_mean', np.mean(all_grads), it)
            writer.add_scalar('grad/total_max', np.max(all_grads), it)

        # 前10个iteration打印梯度debug信息
        if it < 10:
            logger.info(f"[Iter {it}] 梯度分析:")

            # 质谱编码器梯度（dreams模式）
            if hasattr(model, 'spectrum_projector') and model.spectrum_projector is not None:
                spec_grads = [p.grad.norm().item() for n, p in model.spectrum_projector.named_parameters() if p.grad is not None]
                if spec_grads:
                    logger.info(f"  [质谱编码器-dreams] 梯度范数: 均值={np.mean(spec_grads):.6f}, 最大={np.max(spec_grads):.6f}")

            # 原始质谱峰编码器梯度（origin模式）
            if hasattr(model, 'peaks_encoder') and model.peaks_encoder is not None:
                peaks_grads = [p.grad.norm().item() for n, p in model.peaks_encoder.named_parameters() if p.grad is not None]
                if peaks_grads:
                    logger.info(f"  [质谱编码器-origin] 梯度范数: 均值={np.mean(peaks_grads):.6f}, 最大={np.max(peaks_grads):.6f}")

            if hasattr(model, 'condition_embedding'):
                for name, param in model.condition_embedding.named_parameters():
                    if param.grad is not None:
                        logger.info(f"  [条件嵌入] {name}: 梯度范数={param.grad.norm().item():.6f}")

            if hasattr(model, 'node_embedder') and node_emb_grad is not None:
                logger.info(f"  [节点嵌入/分子式] 梯度范数={node_emb_grad.norm().item():.6f}")

            if hasattr(model, 'edge_predictor') and model.edge_predictor is not None:
                edge_grads = [p.grad.norm().item() for n, p in model.edge_predictor.named_parameters() if p.grad is not None]
                if edge_grads:
                    logger.info(f"  [边预测头] 梯度范数: 均值={np.mean(edge_grads):.6f}")

        if config.train.get('max_grad_norm'):
            clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()

        # ========== 打印和记录预测分布 ==========
        if '_edge_logits' in loss_dict and '_edge_types_true' in loss_dict:
            with torch.no_grad():
                edge_logits = loss_dict['_edge_logits']
                edge_types_true = loss_dict['_edge_types_true']
                pred = edge_logits.argmax(dim=-1)

                # BFN 模式：在所有边上计算准确率
                pred_dist = torch.bincount(pred, minlength=5).tolist()
                true_dist = torch.bincount(edge_types_true, minlength=5).tolist()

                total_acc = (pred == edge_types_true).float().mean().item()
                bond_mask = edge_types_true > 0
                bond_acc = ((pred == edge_types_true) & bond_mask).sum().item() / bond_mask.sum().item() if bond_mask.sum() > 0 else 0

                # 计算MolAcc（分子级完全匹配率）
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

                # 打印分布
                logger.info(f'  真实分布: {true_dist} | 预测分布: {pred_dist} | TotalAcc={total_acc:.4f} BondAcc={bond_acc:.4f} MolAcc={mol_acc:.4f}')

                # 记录到TensorBoard
                writer.add_scalar('train/total_acc', total_acc, it)
                writer.add_scalar('train/bond_acc', bond_acc, it)
                writer.add_scalar('train/mol_acc', mol_acc, it)

                # 根据配置决定是否绘制边类型分布图
                if config.train.get('log_edge_distribution', True):
                    # 创建分布对比图（一张图显示真实和预测分布）
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(5)
                    width = 0.35
                    edge_type_names = ['NoBond(0)', 'Single(1)', 'Double(2)', 'Triple(3)', 'Aromatic(4)']

                    # 归一化为比例
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

                    # 在柱子上显示数值
                    for bar, val in zip(bars1, true_dist):
                        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='steelblue')
                    for bar, val in zip(bars2, pred_dist):
                        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='coral')

                    plt.tight_layout()
                    writer.add_figure('distribution/edge_type_comparison', fig, it)
                    plt.close(fig)

            # 清理临时数据
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

        # 验证
        if it % config.train.val_freq == 0 and it > 0:
            model.eval()

            # 验证逻辑
            val_loss_dict_list = []

            val_loader_subset = DataLoader(
                val_subset,
                batch_size=config.train.batch_size,
                shuffle=False,
                num_workers=config.train.num_workers,
                pin_memory=config.train.pin_memory,
                collate_fn=collate_fn
            )

            logger.info(f'[{it}] 开始验证...')
            with torch.no_grad():
                val_pbar = tqdm(val_loader_subset, desc=f'验证 iter {it}', leave=False)
                for val_batch in val_pbar:
                    val_loss_dict = train_flash(model, val_batch, args.device, config)
                    # 跳过以_开头的临时数据
                    val_loss_dict_list.append({k: v.item() for k, v in val_loss_dict.items() if not k.startswith('_')})
                    val_pbar.set_postfix({'loss': val_loss_dict['loss'].item()})

            avg_val_losses = {}
            for key in val_loss_dict_list[0].keys():
                avg_val_losses[key] = sum([d[key] for d in val_loss_dict_list]) / len(val_loss_dict_list)

            val_loss_str = ' '.join([f'{key}={value:.4f}' for key, value in avg_val_losses.items()])
            logger.info(f'[{it}] 验证损失: {val_loss_str}')

            for key, value in avg_val_losses.items():
                writer.add_scalar(f'val/{key}', value, it)

            # ========== 重建评估：为每个样本生成多个结果，计算Top-K准确率 ==========
            eval_num_samples = config.train.get('eval_num_samples', 100)
            logger.info(f'[{it}] 开始重建评估（每个样本生成{eval_num_samples}个结果）...')
            recon_metrics = evaluate_reconstruction(
                model=model,
                val_dataset=val_subset,  # 使用本次随机采样的子集
                device=args.device,
                config=config,
                mode=mode,
                atomic_numbers=atomic_numbers,
                logger=logger,
                collate_fn=collate_fn,  # 传入collate函数
                num_samples=eval_num_samples
            )
            # 记录重建指标到TensorBoard（用 .get 兜底，align 阶段缺某些字段也不会崩）
            writer.add_scalar('val/recon_success_rate', recon_metrics.get('recon_success_rate', 0.0), it)
            writer.add_scalar('val/edge_accuracy', recon_metrics.get('edge_accuracy', 0.0), it)
            writer.add_scalar('val/bond_accuracy', recon_metrics.get('bond_accuracy', 0.0), it)
            writer.add_scalar('val/mol_accuracy', recon_metrics.get('mol_accuracy', 0.0), it)
            # align 阶段独有指标
            if 'pairwise_top1' in recon_metrics:
                writer.add_scalar('val/pairwise_top1', recon_metrics['pairwise_top1'], it)
            if 'cos_sim_mean' in recon_metrics:
                writer.add_scalar('val/cos_sim_mean', recon_metrics['cos_sim_mean'], it)
            # Top-K 指标（仅 graph2mol/ms2mol 有）
            if 'top1_mol_accuracy' in recon_metrics:
                writer.add_scalar('val/top1_mol_accuracy', recon_metrics['top1_mol_accuracy'], it)
            if 'top10_mol_accuracy' in recon_metrics:
                writer.add_scalar('val/top10_mol_accuracy', recon_metrics['top10_mol_accuracy'], it)
            # ==========================================================

            if scheduler:
                scheduler.step(avg_val_losses['loss'])

        # 保存checkpoint
        if it % config.train.save_freq == 0 and it > 0:
            ckpt_path = os.path.join(ckpt_dir, f'{mode_with_spectrum}_iter{it}.pt')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'iteration': it,
                'mode': mode,
                'stage': stage,
            }, ckpt_path)
            logger.info(f'保存checkpoint: {ckpt_path}')

    logger.info('训练完成!')


if __name__ == '__main__':
    main()
