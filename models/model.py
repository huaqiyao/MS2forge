"""
FLASH 模型定义（DeniMS 风格三阶段范式）

输入：分子式（formula，硬约束已知原子序列）+ DreaMS 1024-d 嵌入
输出：分子图边类型（BFN 边去噪）

三阶段（由 yaml 的 model.stage 切换；不再有旧的 spectrum_mode 概念）：
  - align     : MS↔Mol 对齐预训练（双塔 InfoNCE，不跑 BFN）
                ms_encoder(raw DreaMS) ⇄ graph_encoder(node_type, halfedge_index) → 256 维公共空间
  - graph2mol : Graph2Mol 预训练（用 graph_encoder(mol) 当条件 y，BFN 还原边）
  - ms2mol    : MS2Mol 微调（用 ms_encoder(raw DreaMS) 当条件 y，BFN 还原边）

关键设计：
  - graph_encoder 的输入只有 (node_type, halfedge_index)，不含 DreaMS → 与 ms_encoder 双塔无信息泄露
  - 三阶段共享 256 维语义空间（contrastive_dim）
  - 分子式（formula = node_type + halfedge_index 全连接结构）全程作为节点硬约束
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from .gnn import NodeEdgeNet2D, FullyConnectedEdgeInit


# 条件类型定义
INSTRUMENT_TYPES = ['Orbitrap', 'QTOF', 'NONE']
IONIZATION_TYPES = ['[M+H]+', '[M+Na]+']

# DreaMS 嵌入固定维度
DREAMS_DIM = 1024


def _cfg_get(cfg, key, default=None):
    """统一从 dict 或 namespace/EasyDict 取属性"""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ============================================================================
# 以下模块**完整逐行复刻**自 DeniMS 官方代码：
#   - Xtoy / Etoy / masked_softmax    ← DeniMS/MS_diffusion/src/models/layers.py
#   - assert_correctly_masked          ← DeniMS/MS_diffusion/src/diffusion/diffusion_utils.py
#   - PlaceHolder                      ← DeniMS/MS_diffusion/src/utils.py
#   - NodeEdgeBlock / XEyTransformerLayer / GraphTransformer_embedding
#                                      ← DeniMS/MS_diffusion/src/models/transformer_model.py
#   - TransformerModel (ms_encoder)    ← DeniMS/model.py
# 用于直接加载 Encoder_Contrastive_FragHub.pth ckpt（FragHub 上百万级训练）。
# 维度约定：DeniMS hidden_dim=512, output_dim=512, 3 层 transformer。
# ============================================================================

import math


def _denims_assert_masked(variable, node_mask):
    """DeniMS diffusion_utils.assert_correctly_masked"""
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


class _DeniMS_Xtoy(nn.Module):
    """DeniMS layers.Xtoy: 节点 → 全局"""
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class _DeniMS_Etoy(nn.Module):
    """DeniMS layers.Etoy: 边 → 全局"""
    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


def _denims_masked_softmax(x, mask, **kwargs):
    """DeniMS layers.masked_softmax"""
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class _DeniMS_PlaceHolder:
    """DeniMS utils.PlaceHolder（仅用 mask 方法）"""
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
        return self


class _DeniMS_NodeEdgeBlock(nn.Module):
    """DeniMS NodeEdgeBlock 完整复刻"""
    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)
        self.y_y = nn.Linear(dy, dy)
        self.x_y = _DeniMS_Xtoy(dx, dy)
        self.e_y = _DeniMS_Etoy(de, dy)
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        _denims_assert_masked(Q, x_mask)
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)
        K = K.unsqueeze(1)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        _denims_assert_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))
        E1 = self.e_mul(E) * e_mask1 * e_mask2
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        Y = Y * (E1 + 1) + E2
        newE = Y.flatten(start_dim=3)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2
        _denims_assert_masked(newE, e_mask1 * e_mask2)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = _denims_masked_softmax(Y, softmax_mask, dim=2)
        V = self.v(X) * x_mask
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)
        weighted_V = weighted_V.flatten(start_dim=2)
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask
        _denims_assert_masked(newX, x_mask)
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)
        return newX, newE, new_y


class _DeniMS_XEyTransformerLayer(nn.Module):
    """DeniMS XEyTransformerLayer 完整复刻"""
    def __init__(self, dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128, dim_ffy=2048,
                 dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = _DeniMS_NodeEdgeBlock(dx, de, dy, n_head)
        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)
        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)
        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))
        ff_X = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        X = self.normX2(X + self.dropoutX3(ff_X))
        ff_E = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        E = self.normE2(E + self.dropoutE3(ff_E))
        ff_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        y = self.norm_y2(y + self.dropout_y3(ff_y))
        return X, E, y


class GraphEncoder(nn.Module):
    """DeniMS GraphTransformer_embedding 完整复刻。

    输入：dense (X[B,N,11], E[B,N,N,5], y[B,1], node_mask[B,N])
    输出：[B, output_dim] graph embedding（mean pool over valid nodes + MLP_out_X）

    DeniMS ckpt 用的维度（已通过 eval_denims_on_msg.py 确认）：
      n_layers=4, input_dims={'X': 11, 'E': 5, 'y': 1},
      hidden_mlp_dims={'X': 256, 'E': 128, 'y': 1},
      hidden_dims={'dx': 256, 'de': 128, 'dy': 1, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1},
      output_dims={'X': 512}
    """
    def __init__(self, n_layers=4,
                 input_dims=None, hidden_mlp_dims=None, hidden_dims=None, output_dims=None,
                 act_fn_in=None, act_fn_out=None):
        super().__init__()
        # 默认按 DeniMS Contrastive_model 的 graph 分支硬编码（与 ckpt 完全对齐）
        if input_dims is None:
            input_dims = {'X': 11, 'E': 5, 'y': 1}
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 1}
        if hidden_dims is None:
            hidden_dims = {'dx': 256, 'de': 128, 'dy': 1, 'n_head': 8,
                           'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1}
        if output_dims is None:
            output_dims = {'X': 512}
        if act_fn_in is None:
            act_fn_in = nn.ReLU()
        if act_fn_out is None:
            act_fn_out = nn.ReLU()

        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in,
        )
        self.tf_layers = nn.ModuleList([
            _DeniMS_XEyTransformerLayer(
                dx=hidden_dims['dx'], de=hidden_dims['de'], dy=hidden_dims['dy'],
                n_head=hidden_dims['n_head'],
                dim_ffX=hidden_dims['dim_ffX'], dim_ffE=hidden_dims['dim_ffE'],
            ) for _ in range(n_layers)
        ])
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], output_dims['X']),
        )

    def forward(self, X, E, y, node_mask):
        """
        X: [B, N, 11], E: [B, N, N, 5], y: [B, 1], node_mask: [B, N]
        返回 [B, output_dim_X]
        """
        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        new_X = self.mlp_in_X(X)

        after_in = _DeniMS_PlaceHolder(X=new_X, E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # mean pool over valid nodes
        mask = node_mask.unsqueeze(-1)
        X_masked = X * mask
        valid_counts = mask.sum(dim=1).clamp(min=1)
        X_mean = X_masked.sum(dim=1) / valid_counts
        return self.mlp_out_X(X_mean)


class MSEncoder(nn.Module):
    """DeniMS TransformerModel 完整复刻（FragHub ckpt 兼容）。

    输入：
      sos: [B, 1, 13]   precursor one-hot (2) + collision_energy_NCE one-hot (11)
      formula_array: [B, 128, 144]  每峰 9 元素 × 16-d 元素计数 sinusoidal 编码
      mask: [B, 129]    True = padding 位置（src_key_padding_mask）

    输出：[B, output_dim]   spectrum embedding

    与 ckpt 完全对齐的默认维度：
      hidden_dim=512, num_transformer_layers=3, nhead=8, output_dim=512
    """
    def __init__(self, dim_sos=13, dim_formula=144, hidden_dim=512,
                 num_transformer_layers=3, nhead=8, output_dim=512,
                 dropout=0.1, input_dropout=0.1, max_len=129):
        super().__init__()
        self.sos_proj = nn.Linear(dim_sos, hidden_dim)
        self.formula_proj = nn.Linear(dim_formula, hidden_dim)
        self.input_dropout = nn.Dropout(p=input_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False,
        )
        # ckpt 里有 self.mlp，虽然 forward 里没用，但加载 strict=True 需要存在
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2048),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        scale = hidden_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(hidden_dim, output_dim))

    def forward(self, sos, formula_array, mask=None):
        sos_proj = self.sos_proj(sos)             # [B, 1, d]
        formula_proj = self.formula_proj(formula_array)  # [B, 128, d]
        tokens = torch.cat([sos_proj, formula_proj], dim=1)  # [B, 129, d]
        tokens = tokens.transpose(0, 1)            # [129, B, d]
        tokens = self.input_dropout(tokens)
        out = self.transformer_encoder(tokens, src_key_padding_mask=mask)
        out = out.transpose(0, 1)                  # [B, 129, d]
        first_token = out[:, 0, :]                 # [B, d]
        return self.norm(first_token) @ self.proj   # [B, output_dim]


# ============================================================================
# 子模块 3：ConditionEmbedding（仪器类型 + ionization 方式，全程保留）
# ============================================================================
class ConditionEmbedding(nn.Module):
    """仪器类型 + ionization 方式的条件嵌入"""

    def __init__(self, num_instrument_types=3, num_ionization_types=2, embed_dim=256):
        super().__init__()
        self.instrument_embedding = nn.Embedding(num_instrument_types, embed_dim)
        self.ionization_embedding = nn.Embedding(num_ionization_types, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, instrument_idx, ionization_idx):
        inst = self.instrument_embedding(instrument_idx)
        ion = self.ionization_embedding(ionization_idx)
        fused = self.fusion(torch.cat([inst, ion], dim=-1))
        return self.norm(fused)


# ============================================================================
# 主类：FLASH（三阶段统一）
# ============================================================================
class FLASH(nn.Module):
    """FLASH 模型（DeniMS 风格三阶段）。

    所有阶段都保留 formula 硬约束（node_type 由分子式给定、不去噪），
    只在 BFN 边去噪上加 256 维语义条件 y（来自 graph_encoder 或 ms_encoder）。
    """

    def __init__(self, config, num_node_types, num_edge_types, atomic_numbers=None):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_edge_classes = num_edge_types + 1
        self.mask_idx = num_edge_types
        self.atomic_numbers = atomic_numbers

        # ---- 阶段（决定整个模型形态）----
        self.stage = getattr(config, 'stage', None)
        if self.stage not in ('align', 'graph2mol', 'ms2mol'):
            raise ValueError(
                f"model.stage 必须是 align / graph2mol / ms2mol 之一，得到 {self.stage!r}"
            )

        # ---- 维度配置 ----
        self.contrastive_dim = int(_cfg_get(config, 'contrastive_dim', 256))
        cond_cfg = _cfg_get(config, 'condition_embedding', {})
        self.condition_dim = int(_cfg_get(cond_cfg, 'embed_dim', 256))
        self.num_instrument_types = len(INSTRUMENT_TYPES)
        self.num_ionization_types = len(IONIZATION_TYPES)

        # GNN 主干配置（仅 graph2mol / ms2mol 用）
        gat_cfg = _cfg_get(config, 'gat', {})
        self.hidden_dim = int(_cfg_get(gat_cfg, 'hidden_dim', 256))
        self.num_layers = int(_cfg_get(gat_cfg, 'num_layers', 4))
        self.dropout = float(_cfg_get(gat_cfg, 'dropout', 0.1))

        # 对称破缺噪声
        self.noise_scale = float(_cfg_get(config, 'noise_scale', 0.0))
        self.noise_scale_inference = 0.0

        # BFN 步数
        flow_cfg = _cfg_get(config, 'flow', {})
        self.default_n_timesteps = int(_cfg_get(flow_cfg, 'n_timesteps', 100))

        # BFN beta1 buffer
        flash_cfg = _cfg_get(config, 'flash', {})
        beta1_val = float(_cfg_get(flash_cfg, 'beta1', 3.0))
        self.register_buffer('beta1', torch.tensor(beta1_val, dtype=torch.float32))

        # ---- 子模块按 stage 实例化 ----
        ms_enc_cfg = _cfg_get(config, 'ms_encoder', {})
        graph_enc_cfg = _cfg_get(config, 'graph_encoder', {})

        # ms_encoder / graph_encoder：仅 align 阶段实例化
        # graph2mol/ms2mol 阶段绝不实时跑 encoder（与 DeniMS finetune_*=False 等价），
        # 只接收 align 阶段预算好的 cond_emb_cached，所以连模块都不需要构造。
        if self.stage == 'align':
            self.ms_encoder = MSEncoder(
                dim_sos=int(_cfg_get(ms_enc_cfg, 'dim_sos', 13)),
                dim_formula=int(_cfg_get(ms_enc_cfg, 'dim_formula', 144)),
                hidden_dim=int(_cfg_get(ms_enc_cfg, 'hidden_dim', 512)),
                num_transformer_layers=int(_cfg_get(ms_enc_cfg, 'num_transformer_layers', 3)),
                nhead=int(_cfg_get(ms_enc_cfg, 'nhead', 8)),
                output_dim=self.contrastive_dim,
                dropout=float(_cfg_get(ms_enc_cfg, 'dropout', 0.1)),
                input_dropout=float(_cfg_get(ms_enc_cfg, 'input_dropout', 0.1)),
                max_len=int(_cfg_get(ms_enc_cfg, 'max_len', 129)),
            )
            self.graph_encoder = GraphEncoder(
                n_layers=int(_cfg_get(graph_enc_cfg, 'n_layers', 4)),
                output_dims={'X': self.contrastive_dim},
            )
        else:
            self.ms_encoder = None
            self.graph_encoder = None

        # 可学习温度（仅 align 训练时有意义；其它阶段不会被用到）
        if self.stage == 'align':
            init_temp = float(_cfg_get(config, 'contrastive_temperature_init', 30.0))
            self.inv_temperature = nn.Parameter(torch.tensor(1.0 / init_temp))
        else:
            self.inv_temperature = None

        # ---- BFN 主干（仅 graph2mol / ms2mol 实例化）----
        if self.stage in ('graph2mol', 'ms2mol'):
            self._build_bfn_backbone(num_node_types, num_edge_types)
        else:
            # align 阶段不实例化 BFN 主干，省显存且明确边界
            self._bfn_built = False

        # 冻结开关（按 stage 处理）
        self.freeze_graph_encoder = bool(_cfg_get(config, 'freeze_graph_encoder', False))
        self.freeze_ms_encoder = bool(_cfg_get(config, 'freeze_ms_encoder', False))
        if self.stage == 'graph2mol' and self.freeze_graph_encoder and self.graph_encoder is not None:
            for p in self.graph_encoder.parameters():
                p.requires_grad = False
        if self.stage == 'ms2mol' and self.freeze_ms_encoder and self.ms_encoder is not None:
            for p in self.ms_encoder.parameters():
                p.requires_grad = False

        self._print_init_summary()

    # --------------------------------------------------------------
    def _build_bfn_backbone(self, num_node_types, num_edge_types):
        """构造 BFN 主干（graph2mol / ms2mol 共用）。

        节点输入 = 节点 one-hot + 256 维条件 y（graph_emb 或 ms_emb）+ condition + time
        """
        # 节点嵌入维度（输入到 node_proj 前的原始 atom 嵌入）
        node_dim = int(getattr(self.config, 'node_dim', 256))
        self.node_dim = node_dim
        self.node_embedder = nn.Linear(num_node_types, node_dim, bias=False)

        # 条件嵌入（仪器/ionization）
        self.condition_embedding = ConditionEmbedding(
            num_instrument_types=self.num_instrument_types,
            num_ionization_types=self.num_ionization_types,
            embed_dim=self.condition_dim,
        )

        # 时间嵌入
        self.time_dim = 64
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # 边状态嵌入：theta（连续概率向量）通过 Linear 映射
        self.edge_theta_proj = nn.Linear(num_edge_types, self.hidden_dim)
        self.edge_state_embedder = nn.Embedding(self.num_edge_classes, self.hidden_dim)

        # x0 预测头
        self.x0_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_edge_types),
        )

        # 节点输入维度：node_emb + cond_emb (256) + condition (256) + time (64)
        node_input_dim = node_dim + self.contrastive_dim + self.condition_dim + self.time_dim
        self.node_proj = nn.Linear(node_input_dim, self.hidden_dim)

        # 边特征初始化
        self.edge_init = FullyConnectedEdgeInit(
            node_dim=self.hidden_dim, edge_dim=self.hidden_dim,
        )
        self.edge_state_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # 主 BFN 主干（NodeEdgeNet2D；与 graph_encoder 内部独立实例）
        self.gnn = NodeEdgeNet2D(
            node_dim=self.hidden_dim, edge_dim=self.hidden_dim,
            num_blocks=self.num_layers, hidden_dim=self.hidden_dim,
            use_gate=True, dropout=self.dropout, cond_dim=0, use_triangle=False,
        )

        self._bfn_built = True

    # --------------------------------------------------------------
    def _print_init_summary(self):
        print(f"[INFO] FLASH 初始化 (stage={self.stage})")
        print(f"  - contrastive_dim = {self.contrastive_dim}")
        print(f"  - num_node_types  = {self.num_node_types}")
        print(f"  - num_edge_types  = {self.num_edge_types}")
        if self.ms_encoder is not None:
            n = sum(p.numel() for p in self.ms_encoder.parameters())
            print(f"  - ms_encoder      : {n/1e6:.2f}M params")
        if self.graph_encoder is not None:
            n = sum(p.numel() for p in self.graph_encoder.parameters())
            print(f"  - graph_encoder   : {n/1e6:.2f}M params")
        if self._bfn_built:
            print(f"  - BFN 主干       : hidden_dim={self.hidden_dim}, layers={self.num_layers}")
            print(f"  - beta1          : {self.beta1.item()}")
        if self.stage == 'align':
            print(f"  - inv_temperature : {1.0/self.inv_temperature.item():.2f}（temperature）")
        if self.stage == 'graph2mol':
            print(f"  - freeze_graph_encoder = {self.freeze_graph_encoder}")
        if self.stage == 'ms2mol':
            print(f"  - freeze_ms_encoder    = {self.freeze_ms_encoder}")

    # ============================================================
    # forward：按 stage 分支
    # ============================================================
    def forward(self, node_types=None, edge_index=None, batch_node=None, batch_edge=None,
                instrument_type_idx=None, ionization_type_idx=None,
                t=None, edge_types_t=None,
                pretrained_embedding=None, edge_theta=None,
                # ---- DeniMS encoder 专用入参 ----
                spec_sos=None, spec_formula_array=None, spec_mask=None,
                dense_X=None, dense_E=None, dense_y=None, dense_node_mask=None,
                # ---- 预算 cond_emb 缓存（graph2mol/ms2mol 必传，禁止实时推理 encoder）----
                cond_emb_cached=None):
        """
        按 stage 分支：
          - align     : 实时双塔 forward → {'ms_emb', 'graph_emb'}
          - graph2mol : **必须传 cond_emb_cached**（来自 zmol cache，align 阶段提前算好）
                        BFN 主干只接收 cond_emb，graph_encoder 此阶段根本不参与 forward
          - ms2mol    : **必须传 cond_emb_cached**（来自 zms cache）
                        BFN 主干只接收 cond_emb，ms_encoder 此阶段根本不参与 forward

        与 DeniMS 的 finetune_ms_encoder=False 行为完全等价：encoder 在 graph2mol/ms2mol 阶段
        被冻结且不实时跑，BFN 主干训练时只看预算好的 cond_emb 和分子图 sparse 信息。
        """
        if self.stage == 'align':
            ms_emb = self.ms_encoder(spec_sos, spec_formula_array, mask=spec_mask)
            graph_emb = self.graph_encoder(dense_X, dense_E, dense_y, dense_node_mask)
            return {'ms_emb': ms_emb, 'graph_emb': graph_emb}

        # graph2mol / ms2mol：cond_emb 必须由 caller 提供（来自缓存）
        if cond_emb_cached is None:
            raise ValueError(
                f"{self.stage} 阶段必须传 cond_emb_cached（来自 align 阶段预算好的 zmol/zms cache）。"
                f"训练时禁止实时推理 encoder。请确保 collate_fn 把 cond_emb 放进 batch。"
            )
        cond_emb = cond_emb_cached.to(node_types.device if node_types is not None else cond_emb_cached.device)

        return self._forward_bfn(
            node_types=node_types, edge_index=edge_index,
            batch_node=batch_node, batch_edge=batch_edge,
            instrument_type_idx=instrument_type_idx,
            ionization_type_idx=ionization_type_idx,
            t=t, edge_types_t=edge_types_t, edge_theta=edge_theta,
            cond_emb=cond_emb,
        )

    # --------------------------------------------------------------
    def _forward_bfn(self, node_types, edge_index, batch_node, batch_edge,
                     instrument_type_idx, ionization_type_idx,
                     t, edge_types_t, edge_theta, cond_emb):
        """graph2mol / ms2mol 共享：BFN 边去噪 forward。

        cond_emb : [batch_size, contrastive_dim]，来自 graph_encoder 或 ms_encoder
        """
        if instrument_type_idx is None or ionization_type_idx is None:
            raise ValueError("BFN forward 需要 instrument_type_idx / ionization_type_idx")
        if t is None:
            raise ValueError("BFN forward 需要 t")

        # 仪器/电离条件
        condition_features = self.condition_embedding(instrument_type_idx, ionization_type_idx)
        # 时间嵌入
        time_features = self.time_embedding(t.unsqueeze(-1))

        # 节点 one-hot + 嵌入
        node_onehot = F.one_hot(node_types, self.num_node_types).float()
        node_emb = self.node_embedder(node_onehot)

        # 对称破缺噪声
        if self.training:
            node_emb = node_emb + torch.randn_like(node_emb) * self.noise_scale
        else:
            node_emb = node_emb + torch.randn_like(node_emb) * self.noise_scale_inference

        # 广播到节点
        node_cond = cond_emb.index_select(0, batch_node)
        node_condition = condition_features.index_select(0, batch_node)
        node_time = time_features.index_select(0, batch_node)

        # 拼接 + 投影
        node_features = torch.cat([node_emb, node_cond, node_condition, node_time], dim=-1)
        node_features = self.node_proj(node_features)

        # 边特征 + theta 状态融合
        edge_features = self.edge_init(node_features, edge_index)
        if edge_theta is not None:
            edge_state_emb = self.edge_theta_proj(edge_theta)
        else:
            edge_state_emb = self.edge_state_embedder(edge_types_t)
        edge_features = self.edge_state_fusion(torch.cat([edge_features, edge_state_emb], dim=-1))

        # GNN
        node_features, edge_features = self.gnn(node_features, edge_features, edge_index)

        # x0 预测头 → softmax 概率分布
        x0_logits = self.x0_predictor(edge_features)
        return F.softmax(x0_logits, dim=-1)

    # ============================================================
    # InfoNCE 对比损失（align 阶段）
    # ============================================================
    def compute_contrastive_loss(self, ms_emb, graph_emb, positive_mask=None):
        """对应 DeniMS 的双向 InfoNCE：CLIP 风格。

        Args:
            ms_emb / graph_emb : [B, contrastive_dim]
            positive_mask : 可选 [B, B] {0,1} 矩阵
                mask[i,j]=1 表示 (ms_i, graph_j) 是正样本对。
                None 时退化为 CLIP 单正样本（对角线）。
                启用 multi-positive 时由 train_flash 按 batch 内 SMILES 构造（同 SMILES → 1）。

        实现参考：
          - 单正样本：标准 CLIP / DeniMS contrastive
          - 多正样本：SupContrast (HobbitLong/SupContrast losses.py:SupConLoss)
            log_prob = logit - log(sum_neg_exp); loss = -mean(mask * log_prob / mask.sum(1))
        """
        ms = F.normalize(ms_emb, dim=-1)
        gr = F.normalize(graph_emb, dim=-1)

        # clamp inv_temperature ≥ 1/200 → temperature ≤ 200
        with torch.no_grad():
            self.inv_temperature.data.clamp_(min=1.0 / 200.0)
        inv_temp = self.inv_temperature
        # logits = sim / inv_temp  (i.e. sim * temperature)
        logits_ms = (ms @ gr.t()) / inv_temp.clamp(min=1e-3)        # [B, B]
        logits_gr = logits_ms.t()

        B = ms.size(0)
        device = ms.device

        if positive_mask is None:
            # 单正样本 CLIP 模式
            labels = torch.arange(B, device=device)
            loss_ms = F.cross_entropy(logits_ms, labels)
            loss_gr = F.cross_entropy(logits_gr, labels)
            return 0.5 * (loss_ms + loss_gr)

        # ====== Multi-positive (SupCon out-form) ======
        # 参考 SupContrast/losses.py:SupConLoss
        # log_prob = logit - log(sum_j exp(logit_ij))   # 分母包含所有 j（包含正样本）
        # loss = -mean over i of [ (1/|P(i)|) * sum_{j in P(i)} log_prob_ij ]
        mask = positive_mask.float().to(device)

        def _sup_loss(logits):
            # 数值稳定
            logits = logits - logits.max(dim=1, keepdim=True).values.detach()
            exp = logits.exp()
            log_prob = logits - exp.sum(dim=1, keepdim=True).log()
            n_pos = mask.sum(dim=1).clamp(min=1.0)
            return -(mask * log_prob).sum(dim=1).div(n_pos).mean()

        loss_ms = _sup_loss(logits_ms)
        loss_gr = _sup_loss(logits_gr)
        return 0.5 * (loss_ms + loss_gr)

    def compute_mixup_contrastive_loss(self, ms_emb, graph_emb, alpha=1.0):
        """MixCo 风格的 mixup 对比损失（在嵌入空间做 mixup）。

        参考：Lee-Gihun/MixCo-Mixup-Contrast simclr/pretrain.py:_mix_step
        在 batch 内随机配对、按 lam ~ Beta(alpha, alpha) 混合 ms_emb，
        然后让 mixed_ms 对 原 graph_emb 的相似度分布近似 [lam*one_hot(i), (1-lam)*one_hot(j)]。

        Args:
            ms_emb / graph_emb : [B, contrastive_dim]，**已归一化前的** raw（内部归一化）
            alpha : Beta 分布参数，alpha=0 关闭
        Returns:
            mixup loss 标量
        """
        if alpha <= 0:
            return torch.zeros((), device=ms_emb.device)
        B = ms_emb.size(0)
        if B < 2:
            return torch.zeros((), device=ms_emb.device)

        device = ms_emb.device
        # 随机 permutation
        perm = torch.randperm(B, device=device)

        # 在 raw embedding 空间做 mixup，再归一化 → 对应 MixCo 的 "embedding-level mixup"
        # alpha 用 Beta(alpha, alpha) 采样 lam ∈ [0,1]
        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
        lam = lam.view(B, 1)

        ms_mix = lam * ms_emb + (1 - lam) * ms_emb[perm]                    # [B, D]
        gr_mix = lam * graph_emb + (1 - lam) * graph_emb[perm]

        ms_mix_n = F.normalize(ms_mix, dim=-1)
        gr_mix_n = F.normalize(graph_emb, dim=-1)                            # 拿原 graph 做 anchor 集合

        with torch.no_grad():
            self.inv_temperature.data.clamp_(min=1.0 / 200.0)
        inv_temp = self.inv_temperature.clamp(min=1e-3)
        logits_a = (ms_mix_n @ gr_mix_n.t()) / inv_temp                      # [B, B]

        # soft label：对样本 i，target 分布 = lam * one_hot(i) + (1-lam) * one_hot(perm[i])
        soft = torch.zeros(B, B, device=device)
        idx = torch.arange(B, device=device)
        soft[idx, idx] = lam.view(-1)
        soft[idx, perm] = soft[idx, perm] + (1 - lam.view(-1))

        # SoftCrossEntropy（参考 mixco/simclr/loss/mix_loss.py）
        log_probs = F.log_softmax(logits_a, dim=1)
        loss_a = -(soft * log_probs).sum(dim=1).mean()

        # 对称：对 mix graph 同样做一次
        gr_mix_n2 = F.normalize(gr_mix, dim=-1)
        ms_anchor = F.normalize(ms_emb, dim=-1)
        logits_b = (gr_mix_n2 @ ms_anchor.t()) / inv_temp
        log_probs_b = F.log_softmax(logits_b, dim=1)
        loss_b = -(soft * log_probs_b).sum(dim=1).mean()

        return 0.5 * (loss_a + loss_b)

    # ============================================================
    # 离散 BFN 工具：贝叶斯更新 + 损失 + 采样
    # ============================================================
    def discrete_bayesian_update(self, t, edge_types_true, batch_edge):
        """前向带噪：从 one-hot 真值生成 BFN 后验 theta_t"""
        K = self.num_edge_types
        beta1 = self.beta1
        one_hot_x = F.one_hot(edge_types_true, K).float()
        t_edge = t[batch_edge]
        beta = beta1 * (t_edge ** 2)

        mean = beta.unsqueeze(-1) * (K * one_hot_x - 1)
        std = (beta * K).sqrt().unsqueeze(-1)
        eps = torch.randn_like(mean)
        y = mean + std * eps
        return F.softmax(y, dim=-1)

    def compute_bfn_loss(self, e_hat, edge_types_true, t, batch_edge):
        """L = K * beta1 * t * ||e_x - e_hat||²（有键边按密度加权）"""
        losses = {}
        K = self.num_edge_types
        one_hot_x = F.one_hot(edge_types_true, K).float()
        t_edge = t[batch_edge]
        mse_per_edge = ((one_hot_x - e_hat) ** 2).sum(dim=-1)
        bfn_loss_per_edge = K * self.beta1 * t_edge * mse_per_edge

        has_bond = (edge_types_true > 0).float()
        bond_ratio = has_bond.mean()
        weights = 1.0 + (1.0 - bond_ratio) * has_bond
        bfn_loss = (bfn_loss_per_edge * weights).mean()

        losses['bfn_loss'] = bfn_loss.detach()
        losses['total'] = bfn_loss
        losses['_edge_logits'] = e_hat.detach()
        return losses

    @torch.no_grad()
    def sample_bfn(self, node_types, edge_index, batch_node, batch_edge,
                   instrument_type_idx, ionization_type_idx,
                   pretrained_embedding=None,
                   n_timesteps=None, return_trajectory=False, disable_tqdm=False,
                   # ---- 预算好的 cond_emb（graph2mol/ms2mol 必传；align 不调用此函数）----
                   cond_emb_cached=None):
        """BFN 采样：从均匀先验出发迭代 N 步。

        graph2mol/ms2mol 阶段都需要 cond_emb_cached（来自 align 阶段产出的 zmol/zms cache）。
        encoder 不参与采样（与训练一致）。align 阶段不应调用此函数。
        """
        if self.stage == 'align':
            raise RuntimeError("align 阶段没有 BFN 主干，无法采样")
        if cond_emb_cached is None:
            raise ValueError(f"{self.stage} 阶段采样必须传 cond_emb_cached")
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        device = node_types.device
        num_edges = edge_index.shape[1]
        batch_size = batch_node.max().item() + 1
        K = self.num_edge_types
        beta1 = self.beta1.to(device)
        N = n_timesteps

        theta = torch.ones(num_edges, K, device=device) / K

        trajectory = [] if return_trajectory else None
        if return_trajectory:
            trajectory.append({
                'step': 0, 't': 0.0,
                'edge_types': theta.argmax(dim=-1).cpu().numpy().copy(),
                'edge_probs': theta.cpu().numpy().copy(),
            })

        iterator = range(1, N + 1) if disable_tqdm else tqdm(
            range(1, N + 1), desc='BFN采样', leave=False, position=2)

        for i in iterator:
            t_val = (i - 1) / N
            t_batch = torch.full((batch_size,), max(t_val, 1e-4), device=device)

            e_hat = self.forward(
                node_types=node_types, edge_index=edge_index,
                batch_node=batch_node, batch_edge=batch_edge,
                instrument_type_idx=instrument_type_idx,
                ionization_type_idx=ionization_type_idx,
                t=t_batch,
                edge_types_t=torch.zeros(num_edges, dtype=torch.long, device=device),
                edge_theta=theta,
                cond_emb_cached=cond_emb_cached,
            )

            alpha_i = beta1 * (2 * i - 1) / (N ** 2)
            mean = alpha_i * (K * e_hat - 1)
            std = (alpha_i * K).sqrt()
            y = mean + torch.randn_like(mean) * std
            theta = F.softmax(theta.log().clamp(min=-20) + y, dim=-1)

            if return_trajectory:
                trajectory.append({
                    'step': i, 't': i / N,
                    'edge_types': theta.argmax(dim=-1).cpu().numpy().copy(),
                    'edge_probs': theta.cpu().numpy().copy(),
                })

        # 最终 t=1 一步
        t_final = torch.ones(batch_size, device=device)
        e_hat_final = self.forward(
            node_types=node_types, edge_index=edge_index,
            batch_node=batch_node, batch_edge=batch_edge,
            instrument_type_idx=instrument_type_idx,
            ionization_type_idx=ionization_type_idx,
            t=t_final,
            edge_types_t=torch.zeros(num_edges, dtype=torch.long, device=device),
            edge_theta=theta,
            cond_emb_cached=cond_emb_cached,
        )
        edge_types = e_hat_final.argmax(dim=-1)

        if return_trajectory:
            trajectory.append({
                'step': N + 1, 't': 1.0,
                'edge_types': edge_types.cpu().numpy().copy(),
                'edge_probs': e_hat_final.cpu().numpy().copy(),
            })
            return edge_types, trajectory
        return edge_types
