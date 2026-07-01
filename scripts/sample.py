"""FLASH 推理评估脚本（ms2mol 阶段专用）

任务：MS + formula → 分子 2D 结构（边类型预测）

流程：
  1. 加载 MSG split (默认 test)
  2. 用 align ckpt 计算所有 spec 的 Zms 缓存
  3. 加载 ms2mol ckpt
  4. 对每条 spec：用 (Zms, formula 给定的 node_type, halfedge_index)
     调 model.sample_bfn 采样 n_samples 次 → n_samples 个边类型序列
  5. DiffMS-style Top-K 评估：
       a) 每个候选过 valence + connectivity 化学合理性检查（与 DiffMS is_valid 等价）
       b) 候选按出现频率排序去重
       c) top-k 命中 = 真值边类型序列出现在前 k 个去重候选中
       d) 不重建 RDKit 分子（边 tuple 在 formula+node_type 给定下唯一定结构，省 30-50× 时间）

直接运行（默认配置评估 ms2mol_iter72000.pt 在 MSG test 全集，n_samples=100, n_timesteps=20）：
    python scripts/sample.py

命令行覆盖默认值（可选）：
    python scripts/sample.py \\
        --ms2mol_ckpt ./checkpoints/ms2mol/xxx.pt \\
        --align_ckpt  ./checkpoints/Encoder_Contrastive_FragHub.pth \\
        --device cuda \\
        --n_samples 50 --n_timesteps 100 --subset_ratio 0.1
"""
import os
import sys
import argparse
import time
import json
import random
from collections import Counter, defaultdict

sys.path.append('.')

import torch
import yaml
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.dataset import DiffMSMSGDataset, ensure_cond_emb_cache
from utils.transforms import make_msg_diffms_collate_with_cache
from models.model import FLASH


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser('FLASH ms2mol 推理评估')
    p.add_argument('--config',        type=str, default='./configs/sample.yml')
    p.add_argument('--ms2mol_ckpt',   type=str, default=None,
                   help='ms2mol ckpt 路径（默认读 cfg.ckpt.ms2mol = checkpoints/ms2mol/ms2mol_iter72000.pt）')
    p.add_argument('--align_ckpt',    type=str, default=None,
                   help='align ckpt 路径（默认读 cfg.ckpt.align = Encoder_Contrastive_FragHub.pth）')
    p.add_argument('--device',        type=str, default='cuda',
                   help='auto / cuda / cpu。auto = cuda 可用就用 cuda，否则 cpu')
    p.add_argument('--split',         type=str, default=None,
                   help='评估 split（默认 yaml 中 test）')
    p.add_argument('--batch_size',    type=int, default=None)
    p.add_argument('--n_samples',     type=int, default=None,
                   help='每谱采样次数（默认 yaml 中 100；设为 1 即单次采样）')
    p.add_argument('--n_timesteps',   type=int, default=None,
                   help='BFN 采样步数（默认 yaml 中 20）')
    p.add_argument('--subset_ratio',  type=float, default=None,
                   help='评估前 ratio 比例（默认 yaml 中 1.0 = 全集）')
    p.add_argument('--seed',          type=int, default=None)
    p.add_argument('--topk',          type=str, default=None,
                   help='top-k 列表，逗号分隔。默认 yaml 中 [1,5,10]')
    p.add_argument('--output_dir',    type=str, default=None)
    p.add_argument('--save_per_spec', action='store_true',
                   help='保存每条 spec 的预测/真值张量')
    p.add_argument('--inner_chunk',  type=int, default=None,
                   help='单次 sample_bfn 的最大复制次数（OOM 保护）。默认 yaml 中 25')
    p.add_argument('--no_valid_check', action='store_true',
                   help='关闭 DiffMS 风格化学合理性检查（仅用边 tuple 频次排序）')
    return p.parse_args()


# =====================================================================
# DiffMS-style 化学合理性检查（不重建分子，纯张量计算）
# =====================================================================
# NEO BFN 边类型编码：0=NoBond, 1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC
# 把芳香键当作单键计入价态（与 RDKit 默认行为一致：芳香环里每个原子贡献 ~1.5 键，
# 但 RDKit valence check 本质看键数计数，对芳香原子允许 +1 的 aromatic correction）
_BOND_VALENCE = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)  # 索引同 halfedge_type

# NEO BFN 节点类型索引: [B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16),
#                       Cl(17), As(33), Se(34), Br(35), I(53), MASK]
# 各原子最大允许价态（DiffMS 也用类似规则，标准化合价 + 部分宽容）
_MAX_VALENCE = torch.tensor([
    3,   # B
    4,   # C
    5,   # N（含 nitro 等高价态特例）
    2,   # O
    1,   # F
    4,   # Si
    5,   # P
    6,   # S
    1,   # Cl
    5,   # As
    6,   # Se
    1,   # Br
    7,   # I
    99,  # MASK 不查
], dtype=torch.long)


def is_chem_valid(node_type, halfedge_index, halfedge_type):
    """快速 valence + connectivity 检查（取代 RDKit 重建）

    Args:
        node_type: [N] long, NEO BFN 索引
        halfedge_index: [2, M] long, 半边（每条键只占一行）
        halfedge_type: [M] long, 0..4

    Returns:
        bool: True = 化学合理, False = 违反价态或不连通
    """
    N = int(node_type.size(0))
    if N == 0:
        return False
    # ---- 1. valence ----
    valence_per_edge = _BOND_VALENCE[halfedge_type.long()]
    atom_valence = torch.zeros(N, dtype=torch.long)
    src, dst = halfedge_index[0].long(), halfedge_index[1].long()
    atom_valence.index_add_(0, src, valence_per_edge)
    atom_valence.index_add_(0, dst, valence_per_edge)
    max_v = _MAX_VALENCE[node_type.long()]
    if (atom_valence > max_v).any().item():
        return False
    # ---- 2. connectivity (并查集) ----
    # 仅考虑实际存在的键（halfedge_type > 0）
    real_mask = halfedge_type > 0
    real_src = src[real_mask].tolist()
    real_dst = dst[real_mask].tolist()
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, e in zip(real_src, real_dst):
        rs, re_ = find(s), find(e)
        if rs != re_:
            parent[rs] = re_
    roots = set(find(i) for i in range(N))
    return len(roots) == 1


# =====================================================================
# DiffMS-style Top-K：边类型序列层面 + 化学合理性过滤
# =====================================================================
def topk_hit_for_mol(pred_seqs, true_seq, topk_list,
                     node_type=None, halfedge_index=None, do_valid_check=True):
    """
    pred_seqs: List[Tensor]，n_samples 个边类型序列（每个 shape=[num_halfedges]）
    true_seq:  Tensor[num_halfedges]，真实边类型
    topk_list: List[int]，要计算的 k 值
    node_type, halfedge_index: 用于化学合理性检查（不传则跳过）
    do_valid_check: True 时按 DiffMS 流程过滤化学不合理的候选

    返回 (hits dict, n_valid)
    DiffMS 协议：
      - 候选先过 is_valid 化学合理性检查 → 不合法的丢弃
      - 剩余候选按出现频率排序去重
      - 前 k 个去重候选里有真值即命中
    """
    # 1) 过滤无效候选
    if do_valid_check and node_type is not None and halfedge_index is not None:
        valid_seqs = [p for p in pred_seqs
                      if is_chem_valid(node_type, halfedge_index, p)]
    else:
        valid_seqs = pred_seqs
    n_valid = len(valid_seqs)

    # 2) 转 tuple 当"分子身份"（在 formula 给定 + node_type 固定下，边 tuple 唯一定结构）
    pred_tuples = [tuple(p.tolist()) for p in valid_seqs]
    true_tuple = tuple(true_seq.tolist())

    # 3) 按频次排序（同频按插入序）
    counter = Counter(pred_tuples)
    sorted_unique = [t for t, _ in counter.most_common()]

    hits = {}
    for k in topk_list:
        hits[k] = true_tuple in sorted_unique[:k]
    return hits, n_valid


# =====================================================================
# 主评估
# =====================================================================
def main():
    args = parse_args()

    # ---- 载入配置 ----
    cfg = EasyDict(yaml.safe_load(open(args.config)))

    # 命令行覆盖
    if args.split is not None:        cfg.evaluate.split = args.split
    if args.batch_size is not None:   cfg.evaluate.batch_size = args.batch_size
    if args.n_samples is not None:    cfg.evaluate.n_samples = args.n_samples
    if args.n_timesteps is not None:  cfg.evaluate.n_timesteps = args.n_timesteps
    if args.subset_ratio is not None: cfg.evaluate.subset_ratio = args.subset_ratio
    if args.seed is not None:         cfg.evaluate.seed = args.seed
    if args.output_dir is not None:   cfg.evaluate.output_dir = args.output_dir
    if args.topk is not None:
        cfg.evaluate.topk = [int(x) for x in args.topk.split(',')]
    if args.inner_chunk is not None:
        cfg.evaluate.inner_chunk = args.inner_chunk
    cfg.evaluate.inner_chunk = int(cfg.evaluate.get('inner_chunk', cfg.evaluate.n_samples))
    if args.no_valid_check:
        cfg.evaluate.valid_check = False

    ms2mol_ckpt = args.ms2mol_ckpt or cfg.ckpt.ms2mol
    align_ckpt  = args.align_ckpt  or cfg.ckpt.align

    assert os.path.exists(ms2mol_ckpt), f'ms2mol ckpt 不存在: {ms2mol_ckpt}'
    assert os.path.exists(align_ckpt),  f'align ckpt 不存在: {align_ckpt}'

    # 强制 ms2mol 阶段
    cfg.model.stage = 'ms2mol'

    seed = int(cfg.evaluate.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 设备自动检测
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print('  [警告] --device cuda 但 CUDA 不可用，自动改用 cpu')
            device = 'cpu'
    os.makedirs(cfg.evaluate.output_dir, exist_ok=True)

    print('=' * 70)
    print('FLASH ms2mol 推理评估（DiffMS-style Top-K）')
    print('=' * 70)
    print(f'  ms2mol_ckpt    : {ms2mol_ckpt}')
    print(f'  align_ckpt     : {align_ckpt}')
    print(f'  device         : {device}')
    print(f'  split          : {cfg.evaluate.split}')
    print(f'  batch_size     : {cfg.evaluate.batch_size}')
    print(f'  n_samples/谱   : {cfg.evaluate.n_samples}')
    print(f'  inner_chunk    : {cfg.evaluate.inner_chunk}（每次 sample_bfn 复制次数）')
    print(f'  n_timesteps    : {cfg.evaluate.n_timesteps}')
    print(f'  subset_ratio   : {cfg.evaluate.subset_ratio}')
    print(f'  topk           : {cfg.evaluate.topk}')
    print(f'  valid_check    : {cfg.evaluate.get("valid_check", True)}')
    print(f'  seed           : {seed}')
    print('=' * 70)

    # ---- 数据集 ----
    print('\n[1/4] 加载 MSG 数据集 ...')
    ds = DiffMSMSGDataset(root=cfg.dataset.root)
    split_name = cfg.evaluate.split
    assert split_name in ds.subsets, f'split={split_name} 不在 {list(ds.subsets.keys())}'

    full_subset = ds.subsets[split_name]
    full_indices = list(full_subset.indices)
    total_n = len(full_indices)

    if cfg.evaluate.subset_ratio < 1.0:
        n_keep = max(1, int(total_n * cfg.evaluate.subset_ratio))
        sel_indices = random.Random(seed).sample(full_indices, n_keep)
        print(f'  subset_ratio={cfg.evaluate.subset_ratio}: 随机抽 {n_keep}/{total_n}')
        eval_subset = Subset(ds, sel_indices)
    else:
        print(f'  使用全部 {split_name}: {total_n} 条')
        eval_subset = full_subset

    # ---- Zms cache ----
    print(f'\n[2/4] 构建/加载 Zms cache (用 align_ckpt) ...')
    zms_cache = ensure_cond_emb_cache(
        stage='ms2mol',
        align_ckpt_path=align_ckpt,
        msg_root=cfg.dataset.root,
        cache_dir=getattr(cfg.dataset, 'cache_dir', './data/cache'),
        device=device,
        batch_size=64,
    )
    print(f'  zms_cache: {len(zms_cache)} 条 spec embedding')

    collate_fn = make_msg_diffms_collate_with_cache(zms_cache)
    loader = DataLoader(
        eval_subset,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- 模型 ----
    print(f'\n[3/4] 加载 ms2mol ckpt ...')
    model = FLASH(
        cfg.model,
        num_node_types=len(cfg.chem.atomic_numbers) + 1,   # +1 表示 mask
        num_edge_types=len(cfg.chem.mol_bond_types) + 1,   # +1 表示 NoBond
    ).to(device)
    try:
        sd = torch.load(ms2mol_ckpt, map_location=device, weights_only=False)
    except TypeError:
        # 老版本 PyTorch (<2.0) 不支持 weights_only 参数
        sd = torch.load(ms2mol_ckpt, map_location=device)
    state = sd.get('model', sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  ms2mol ckpt 加载: missing={len(missing)}, unexpected={len(unexpected)}')
    if missing:
        print(f'    [警告] missing keys (前 5): {missing[:5]}')
    if unexpected:
        print(f'    [警告] unexpected keys (前 5): {unexpected[:5]}')
    model.eval()

    # ---- 评估 ----
    print(f'\n[4/4] 开始评估 ...')
    topk_list = list(cfg.evaluate.topk)
    n_samples = int(cfg.evaluate.n_samples)
    n_timesteps = int(cfg.evaluate.n_timesteps)
    inner_chunk = int(cfg.evaluate.inner_chunk)
    if inner_chunk > n_samples:
        inner_chunk = n_samples
    # 把 n_samples 拆成若干 chunk（OOM 保护）
    chunk_sizes = []
    remain = n_samples
    while remain > 0:
        c = min(inner_chunk, remain)
        chunk_sizes.append(c)
        remain -= c

    # 累计
    topk_correct = {k: 0 for k in topk_list}
    total_mols = 0
    n_valid_total = 0   # 通过化学合理性检查的候选总数（统计用）
    # 边类型统计（细粒度）
    total_edges = 0
    edge_correct = 0      # all edge 位置（含 NoBond）
    bond_total = 0
    bond_correct = 0      # bond_mask>0 位置（仅真实键）

    # 可选保存
    per_spec_records = []

    t_start = time.time()
    pbar = tqdm(loader, desc=f'eval {split_name}', total=len(loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            batch = batch.to(device)

            if not hasattr(batch, 'cond_emb_cached') or batch.cond_emb_cached is None:
                # 整个 batch 没有可用 Zms（不太可能；防御）
                continue

            cond_emb = batch.cond_emb_cached.to(device)                 # [B, 512]
            bsize = cond_emb.size(0)

            # 仪器 / ionization 条件
            instrument_idx = (batch.instrument_type_idx_batch.to(device)
                              if getattr(batch, 'instrument_type_idx_batch', None) is not None
                              else torch.zeros(bsize, dtype=torch.long, device=device))
            ionization_idx = (batch.ionization_type_idx_batch.to(device)
                              if getattr(batch, 'ionization_type_idx_batch', None) is not None
                              else torch.zeros(bsize, dtype=torch.long, device=device))

            edge_true = batch.halfedge_type        # [E]
            num_nodes = batch.node_type.shape[0]
            num_edges = batch.halfedge_index.shape[1]

            # ===== 分 chunk 采样，累计到 preds_per_sample =====
            preds_per_sample = []   # 总共会有 n_samples 个 [num_edges] tensor
            for chunk_n in chunk_sizes:
                node_types_exp = batch.node_type.repeat(chunk_n)
                edge_index_exp = batch.halfedge_index.repeat(1, chunk_n)
                for s in range(1, chunk_n):
                    edge_index_exp[:, s * num_edges:(s + 1) * num_edges] += s * num_nodes
                batch_node_exp = torch.cat([batch.node_type_batch + s * bsize
                                             for s in range(chunk_n)])
                batch_edge_exp = torch.cat([batch.halfedge_type_batch + s * bsize
                                             for s in range(chunk_n)])
                instrument_exp = instrument_idx.repeat(chunk_n)
                ionization_exp = ionization_idx.repeat(chunk_n)
                cond_emb_exp   = cond_emb.repeat(chunk_n, 1)

                pred_chunk = model.sample_bfn(
                    node_types=node_types_exp,
                    edge_index=edge_index_exp,
                    batch_node=batch_node_exp,
                    batch_edge=batch_edge_exp,
                    instrument_type_idx=instrument_exp,
                    ionization_type_idx=ionization_exp,
                    cond_emb_cached=cond_emb_exp,
                    n_timesteps=n_timesteps,
                    disable_tqdm=True,
                )
                # 拆成 chunk_n 份
                for s in range(chunk_n):
                    preds_per_sample.append(
                        pred_chunk[s * num_edges:(s + 1) * num_edges].cpu()
                    )
                del pred_chunk
                if device != 'cpu':
                    torch.cuda.empty_cache()
            assert len(preds_per_sample) == n_samples

            # 用第 1 次采样算细粒度边/键准确率（与训练 evaluate_reconstruction 对齐）
            edge_true_cpu = edge_true.cpu()
            pred0 = preds_per_sample[0]
            edge_correct += (pred0 == edge_true_cpu).sum().item()
            total_edges  += edge_true_cpu.numel()
            bond_mask = edge_true_cpu > 0
            bond_correct += ((pred0 == edge_true_cpu) & bond_mask).sum().item()
            bond_total   += bond_mask.sum().item()

            # ===== 按分子拆分，每个分子算 top-k =====
            num_mols = int(batch.halfedge_type_batch.max().item()) + 1
            edge_batch_cpu = batch.halfedge_type_batch.cpu()

            for mol_idx in range(num_mols):
                mol_mask = (edge_batch_cpu == mol_idx)
                if mol_mask.sum().item() == 0:
                    continue
                true_seq = edge_true_cpu[mol_mask]
                pred_seqs = [p[mol_mask] for p in preds_per_sample]

                # 取该分子对应的 node_type 和 halfedge_index（用于 valence + connectivity 检查）
                node_mask_mol = (batch.node_type_batch.cpu() == mol_idx)
                node_type_mol = batch.node_type.cpu()[node_mask_mol]
                # 找该分子节点在全局的 offset，把 halfedge_index 转回局部索引
                node_idx_global = node_mask_mol.nonzero(as_tuple=True)[0]
                if node_idx_global.numel() == 0:
                    continue
                node_offset = int(node_idx_global[0].item())
                halfedge_index_mol = batch.halfedge_index.cpu()[:, mol_mask] - node_offset

                hits, n_valid_local = topk_hit_for_mol(
                    pred_seqs, true_seq, topk_list,
                    node_type=node_type_mol,
                    halfedge_index=halfedge_index_mol,
                    do_valid_check=cfg.evaluate.get('valid_check', True),
                )
                for k in topk_list:
                    if hits[k]:
                        topk_correct[k] += 1
                total_mols += 1
                n_valid_total += n_valid_local

                if args.save_per_spec or cfg.evaluate.get('save_per_spec', False):
                    per_spec_records.append({
                        'mol_local_idx': mol_idx,
                        'batch_idx': batch_idx,
                        'spec_id': batch.mol_ids[mol_idx] if hasattr(batch, 'mol_ids') else None,
                        'smiles': batch.smiles[mol_idx] if hasattr(batch, 'smiles') else None,
                        'true_edge': true_seq.tolist(),
                        'pred_edges': [p.tolist() for p in pred_seqs],
                        'hits': hits,
                        'n_valid': n_valid_local,
                    })

            # 实时进度
            postfix = {f'top{k}': f'{topk_correct[k]/max(1,total_mols)*100:.2f}%'
                       for k in topk_list}
            postfix['n_mols'] = total_mols
            pbar.set_postfix(postfix)

    elapsed = time.time() - t_start

    # ===== 汇总 =====
    print('\n' + '=' * 70)
    print('★ DiffMS-style Top-K 结果（边类型序列层面）')
    print('=' * 70)
    print(f'  split        : {split_name}')
    print(f'  评估 spec 数 : {total_mols}')
    print(f'  n_samples/谱 : {n_samples}')
    print(f'  n_timesteps  : {n_timesteps}')
    print(f'  耗时         : {elapsed:.1f}s ({elapsed/max(1,total_mols)*1000:.0f}ms/谱)')
    print('-' * 70)
    for k in topk_list:
        acc = topk_correct[k] / max(1, total_mols)
        print(f'  top-{k:<3}      : {acc*100:.4f}%  ({topk_correct[k]}/{total_mols})')
    print('-' * 70)
    print(f'  edge_acc     : {edge_correct/max(1,total_edges)*100:.4f}%   '
          f'(含 NoBond，{edge_correct}/{total_edges})')
    print(f'  bond_acc     : {bond_correct/max(1,bond_total)*100:.4f}%   '
          f'(仅真实键，{bond_correct}/{bond_total})')
    if cfg.evaluate.get('valid_check', True):
        avg_valid = n_valid_total / max(1, total_mols)
        print(f'  化学合理候选 : {avg_valid:.1f} / {n_samples}（每谱平均通过 valence+connectivity 检查的候选数）')
    print('=' * 70)

    # ===== 保存结果 =====
    out_dir = cfg.evaluate.output_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = (f'{split_name}'
           f'_n{n_samples}'
           f'_T{n_timesteps}'
           f'_r{cfg.evaluate.subset_ratio}'
           f'_seed{seed}')
    summary = {
        'split': split_name,
        'total_mols': total_mols,
        'n_samples': n_samples,
        'n_timesteps': n_timesteps,
        'subset_ratio': cfg.evaluate.subset_ratio,
        'seed': seed,
        'elapsed_sec': elapsed,
        'ms2mol_ckpt': ms2mol_ckpt,
        'align_ckpt': align_ckpt,
        'topk': {f'top_{k}': topk_correct[k] / max(1, total_mols) for k in topk_list},
        'topk_correct': {f'top_{k}': topk_correct[k] for k in topk_list},
        'edge_accuracy': edge_correct / max(1, total_edges),
        'bond_accuracy': bond_correct / max(1, bond_total),
        'edge_correct': edge_correct,
        'edge_total': total_edges,
        'bond_correct': bond_correct,
        'bond_total': bond_total,
        'n_valid_per_spec_avg': n_valid_total / max(1, total_mols),
        'valid_check': bool(cfg.evaluate.get('valid_check', True)),
    }
    summary_path = os.path.join(out_dir, f'summary_{tag}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n汇总写入: {summary_path}')

    if args.save_per_spec or cfg.evaluate.get('save_per_spec', False):
        import pickle
        records_path = os.path.join(out_dir, f'per_spec_{tag}.pkl')
        with open(records_path, 'wb') as f:
            pickle.dump(per_spec_records, f)
        print(f'每谱明细写入: {records_path}')


if __name__ == '__main__':
    main()
