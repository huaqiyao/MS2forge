"""MS2Forge module."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from .gnn import NodeEdgeNet2D, FullyConnectedEdgeInit



INSTRUMENT_TYPES = ['Orbitrap', 'QTOF', 'NONE']
IONIZATION_TYPES = ['[M+H]+', '[M+Na]+']


DREAMS_DIM = 1024


def _cfg_get(cfg, key, default=None):
    """_cfg_get implementation."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ============================================================================




# ============================================================================

import math


def _ms2forge_assert_masked(variable, node_mask):
    """_ms2forge_assert_masked implementation."""
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


class _MS2Forge_Xtoy(nn.Module):
    """_MS2Forge_Xtoy implementation."""
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


class _MS2Forge_Etoy(nn.Module):
    """_MS2Forge_Etoy implementation."""
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


def _ms2forge_masked_softmax(x, mask, **kwargs):
    """_ms2forge_masked_softmax implementation."""
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class _MS2Forge_PlaceHolder:
    """_MS2Forge_PlaceHolder implementation."""
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


class _MS2Forge_NodeEdgeBlock(nn.Module):
    """_MS2Forge_NodeEdgeBlock implementation."""
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
        self.x_y = _MS2Forge_Xtoy(dx, dy)
        self.e_y = _MS2Forge_Etoy(de, dy)
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
        _ms2forge_assert_masked(Q, x_mask)
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)
        K = K.unsqueeze(1)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        _ms2forge_assert_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))
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
        _ms2forge_assert_masked(newE, e_mask1 * e_mask2)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = _ms2forge_masked_softmax(Y, softmax_mask, dim=2)
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
        _ms2forge_assert_masked(newX, x_mask)
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)
        return newX, newE, new_y


class _MS2Forge_XEyTransformerLayer(nn.Module):
    """_MS2Forge_XEyTransformerLayer implementation."""
    def __init__(self, dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128, dim_ffy=2048,
                 dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = _MS2Forge_NodeEdgeBlock(dx, de, dy, n_head)
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
    """GraphEncoder implementation."""
    def __init__(self, n_layers=4,
                 input_dims=None, hidden_mlp_dims=None, hidden_dims=None, output_dims=None,
                 act_fn_in=None, act_fn_out=None):
        super().__init__()

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
            _MS2Forge_XEyTransformerLayer(
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
        """forward implementation."""
        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        new_X = self.mlp_in_X(X)

        after_in = _MS2Forge_PlaceHolder(X=new_X, E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
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
    """MSEncoder implementation."""
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

# ============================================================================
class ConditionEmbedding(nn.Module):
    """ConditionEmbedding implementation."""

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

# ============================================================================
class FLASH(nn.Module):
    """FLASH implementation."""

    def __init__(self, config, num_node_types, num_edge_types, atomic_numbers=None):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_edge_classes = num_edge_types + 1
        self.mask_idx = num_edge_types
        self.atomic_numbers = atomic_numbers


        self.stage = getattr(config, 'stage', None)
        if self.stage not in ('graph2mol', 'ms2mol', 'joint'):
            raise ValueError(
                f"model.stage must be one of graph2mol, ms2mol, or joint; received {self.stage!r}"
            )


        self.contrastive_dim = int(_cfg_get(config, 'contrastive_dim', 256))
        cond_cfg = _cfg_get(config, 'condition_embedding', {})
        self.condition_dim = int(_cfg_get(cond_cfg, 'embed_dim', 256))
        self.num_instrument_types = len(INSTRUMENT_TYPES)
        self.num_ionization_types = len(IONIZATION_TYPES)


        gat_cfg = _cfg_get(config, 'gat', {})
        self.hidden_dim = int(_cfg_get(gat_cfg, 'hidden_dim', 256))
        self.num_layers = int(_cfg_get(gat_cfg, 'num_layers', 4))
        self.dropout = float(_cfg_get(gat_cfg, 'dropout', 0.1))


        self.noise_scale = float(_cfg_get(config, 'noise_scale', 0.0))
        self.noise_scale_inference = 0.0


        flow_cfg = _cfg_get(config, 'flow', {})
        self.default_n_timesteps = int(_cfg_get(flow_cfg, 'n_timesteps', 100))

        # BFN beta1 buffer
        flash_cfg = _cfg_get(config, 'flash', {})
        beta1_val = float(_cfg_get(flash_cfg, 'beta1', 3.0))
        self.register_buffer('beta1', torch.tensor(beta1_val, dtype=torch.float32))


        ms_enc_cfg = _cfg_get(config, 'ms_encoder', {})
        graph_enc_cfg = _cfg_get(config, 'graph_encoder', {})







        self.ms_encoder = None
        self.graph_encoder = None
        self.inv_temperature = None


        if self.stage in ('graph2mol', 'ms2mol', 'joint'):
            self._build_bfn_backbone(num_node_types, num_edge_types)
        else:

            self._bfn_built = False

        self._print_init_summary()

    # --------------------------------------------------------------
    def _build_bfn_backbone(self, num_node_types, num_edge_types):
        """_build_bfn_backbone implementation."""

        node_dim = int(getattr(self.config, 'node_dim', 256))
        self.node_dim = node_dim
        self.node_embedder = nn.Linear(num_node_types, node_dim, bias=False)


        self.condition_embedding = ConditionEmbedding(
            num_instrument_types=self.num_instrument_types,
            num_ionization_types=self.num_ionization_types,
            embed_dim=self.condition_dim,
        )


        self.time_dim = 64
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )


        self.edge_theta_proj = nn.Linear(num_edge_types, self.hidden_dim)
        self.edge_state_embedder = nn.Embedding(self.num_edge_classes, self.hidden_dim)


        self.x0_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_edge_types),
        )


        node_input_dim = node_dim + self.contrastive_dim + self.condition_dim + self.time_dim
        self.node_proj = nn.Linear(node_input_dim, self.hidden_dim)



        if self.stage in ('ms2mol', 'joint'):
            self.zms_adapter = nn.Sequential(
                nn.Linear(self.contrastive_dim, self.contrastive_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.contrastive_dim * 2, self.contrastive_dim),
            )
        else:
            self.zms_adapter = None
        self.use_zms_adapter = False


        self.edge_init = FullyConnectedEdgeInit(
            node_dim=self.hidden_dim, edge_dim=self.hidden_dim,
        )
        self.edge_state_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)


        self.gnn = NodeEdgeNet2D(
            node_dim=self.hidden_dim, edge_dim=self.hidden_dim,
            num_blocks=self.num_layers, hidden_dim=self.hidden_dim,
            use_gate=True, dropout=self.dropout, cond_dim=0, use_triangle=False,
        )

        self._bfn_built = True

    # --------------------------------------------------------------
    def _print_init_summary(self):
        print(f"[INFO] FLASH initialization (stage={self.stage})")
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
            print(f"  - BFN backbone       : hidden_dim={self.hidden_dim}, layers={self.num_layers}")
            print(f"  - beta1          : {self.beta1.item()}")

    # ============================================================

    # ============================================================
    def forward(self, node_types=None, edge_index=None, batch_node=None, batch_edge=None,
                instrument_type_idx=None, ionization_type_idx=None,
                t=None, edge_types_t=None,
                pretrained_embedding=None, edge_theta=None,

                spec_sos=None, spec_formula_array=None, spec_mask=None,
                dense_X=None, dense_E=None, dense_y=None, dense_node_mask=None,

                cond_emb_cached=None):
        """forward implementation."""

        if cond_emb_cached is None:
            raise ValueError(
                f"{self.stage} requires cond_emb_cached from an offline Zmol/Zms cache"
            )
        cond_emb = cond_emb_cached.to(
            node_types.device if node_types is not None else cond_emb_cached.device
        )


        self._last_zms_raw = cond_emb
        if getattr(self, 'use_zms_adapter', False) and self.zms_adapter is not None:
            cond_emb = self.zms_adapter(cond_emb)
        self._last_adapter_out = cond_emb

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
        """_forward_bfn implementation."""
        if instrument_type_idx is None or ionization_type_idx is None:
            raise ValueError("BFN forward requires instrument_type_idx and ionization_type_idx")
        if t is None:
            raise ValueError("BFN forward requires t")


        condition_features = self.condition_embedding(instrument_type_idx, ionization_type_idx)

        time_features = self.time_embedding(t.unsqueeze(-1))


        node_onehot = F.one_hot(node_types, self.num_node_types).float()
        node_emb = self.node_embedder(node_onehot)


        if self.training:
            node_emb = node_emb + torch.randn_like(node_emb) * self.noise_scale
        else:
            node_emb = node_emb + torch.randn_like(node_emb) * self.noise_scale_inference


        node_cond = cond_emb.index_select(0, batch_node)
        node_condition = condition_features.index_select(0, batch_node)
        node_time = time_features.index_select(0, batch_node)


        node_features = torch.cat([node_emb, node_cond, node_condition, node_time], dim=-1)
        node_features = self.node_proj(node_features)


        edge_features = self.edge_init(node_features, edge_index)
        if edge_theta is not None:
            edge_state_emb = self.edge_theta_proj(edge_theta)
        else:
            edge_state_emb = self.edge_state_embedder(edge_types_t)
        edge_features = self.edge_state_fusion(torch.cat([edge_features, edge_state_emb], dim=-1))


        if getattr(self, 'capture_layer_h_nodes', False):
            node_features, edge_features, layer_h_nodes = self.gnn(
                node_features, edge_features, edge_index, return_intermediate=True
            )
            self._last_layer_h_nodes = layer_h_nodes
        else:
            node_features, edge_features = self.gnn(node_features, edge_features, edge_index)
            self._last_layer_h_nodes = None


        x0_logits = self.x0_predictor(edge_features)


        self._last_x0_logits = x0_logits
        return F.softmax(x0_logits, dim=-1)

    # ============================================================

    # ============================================================
    def compute_contrastive_loss(self, ms_emb, graph_emb, positive_mask=None):
        """compute_contrastive_loss implementation."""
        ms = F.normalize(ms_emb, dim=-1)
        gr = F.normalize(graph_emb, dim=-1)

        # clamp inv_temperature ≥ 1/200  ->  temperature ≤ 200
        with torch.no_grad():
            self.inv_temperature.data.clamp_(min=1.0 / 200.0)
        inv_temp = self.inv_temperature
        # logits = sim / inv_temp  (i.e. sim * temperature)
        logits_ms = (ms @ gr.t()) / inv_temp.clamp(min=1e-3)        # [B, B]
        logits_gr = logits_ms.t()

        B = ms.size(0)
        device = ms.device

        if positive_mask is None:

            labels = torch.arange(B, device=device)
            loss_ms = F.cross_entropy(logits_ms, labels)
            loss_gr = F.cross_entropy(logits_gr, labels)
            return 0.5 * (loss_ms + loss_gr)

        # ====== Multi-positive (SupCon out-form) ======


        # loss = -mean over i of [ (1/|P(i)|) * sum_{j in P(i)} log_prob_ij ]
        mask = positive_mask.float().to(device)

        def _sup_loss(logits):

            logits = logits - logits.max(dim=1, keepdim=True).values.detach()
            exp = logits.exp()
            log_prob = logits - exp.sum(dim=1, keepdim=True).log()
            n_pos = mask.sum(dim=1).clamp(min=1.0)
            return -(mask * log_prob).sum(dim=1).div(n_pos).mean()

        loss_ms = _sup_loss(logits_ms)
        loss_gr = _sup_loss(logits_gr)
        return 0.5 * (loss_ms + loss_gr)

    def compute_mixup_contrastive_loss(self, ms_emb, graph_emb, alpha=1.0):
        """compute_mixup_contrastive_loss implementation."""
        if alpha <= 0:
            return torch.zeros((), device=ms_emb.device)
        B = ms_emb.size(0)
        if B < 2:
            return torch.zeros((), device=ms_emb.device)

        device = ms_emb.device

        perm = torch.randperm(B, device=device)



        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
        lam = lam.view(B, 1)

        ms_mix = lam * ms_emb + (1 - lam) * ms_emb[perm]                    # [B, D]
        gr_mix = lam * graph_emb + (1 - lam) * graph_emb[perm]

        ms_mix_n = F.normalize(ms_mix, dim=-1)
        gr_mix_n = F.normalize(graph_emb, dim=-1)

        with torch.no_grad():
            self.inv_temperature.data.clamp_(min=1.0 / 200.0)
        inv_temp = self.inv_temperature.clamp(min=1e-3)
        logits_a = (ms_mix_n @ gr_mix_n.t()) / inv_temp                      # [B, B]


        soft = torch.zeros(B, B, device=device)
        idx = torch.arange(B, device=device)
        soft[idx, idx] = lam.view(-1)
        soft[idx, perm] = soft[idx, perm] + (1 - lam.view(-1))


        log_probs = F.log_softmax(logits_a, dim=1)
        loss_a = -(soft * log_probs).sum(dim=1).mean()


        gr_mix_n2 = F.normalize(gr_mix, dim=-1)
        ms_anchor = F.normalize(ms_emb, dim=-1)
        logits_b = (gr_mix_n2 @ ms_anchor.t()) / inv_temp
        log_probs_b = F.log_softmax(logits_b, dim=1)
        loss_b = -(soft * log_probs_b).sum(dim=1).mean()

        return 0.5 * (loss_a + loss_b)

    # ============================================================

    # ============================================================
    def discrete_bayesian_update(self, t, edge_types_true, batch_edge):
        """discrete_bayesian_update implementation."""
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
        """compute_bfn_loss implementation."""
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

                   cond_emb_cached=None,

                   spec_sos=None, spec_formula_array=None, spec_mask=None):
        """sample_bfn implementation."""
        if cond_emb_cached is None:
            raise ValueError(f"{self.stage} sampling requires an offline cond_emb_cached tensor")
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
            range(1, N + 1),
            desc=f'  BFN denoising (T={N} steps)',
            leave=False, position=2, mininterval=0.5,
            unit='step',
        )


        fwd_extra = {'cond_emb_cached': cond_emb_cached}

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
                **fwd_extra,
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


        t_final = torch.ones(batch_size, device=device)
        e_hat_final = self.forward(
            node_types=node_types, edge_index=edge_index,
            batch_node=batch_node, batch_edge=batch_edge,
            instrument_type_idx=instrument_type_idx,
            ionization_type_idx=ionization_type_idx,
            t=t_final,
            edge_types_t=torch.zeros(num_edges, dtype=torch.long, device=device),
            edge_theta=theta,
            **fwd_extra,
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
