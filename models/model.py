"""Permutation-equivariant BFN with shared formula-conditioned FiLM.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, Union
import torch
from torch import nn
from torch.nn import functional as F
from .bfn import MS2Forge as ParentMS2Forge
from .bfn import GraphEncoder, MSEncoder, INSTRUMENT_TYPES, IONIZATION_TYPES
from utils.compat import normalize_model_config

EQ_COND_VARIANT = "eq_cond_formula_film_v1"
ELEMENT_ORDER = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")
FORMULA_DIM = len(ELEMENT_ORDER)
PARENT_ARCHITECTURE_ID = "unordered_bfn_v2"
PARENT_INPUT_PROTOCOL_ID = "EQ_MAIN_FORMULA_V1"
_CLASS_CACHE = {}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(k): _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value

class FormulaCondition(nn.Module):
    """Encode formula-only inputs shared by node and edge FiLM heads."""

    element_order = ELEMENT_ORDER

    def __init__(self, z_dim: int = 512, condition_dim: int = 256):
        super().__init__()
        self.z_dim = int(z_dim)
        self.formula_dim = FORMULA_DIM
        self.condition_dim = int(condition_dim)
        self.input_dim = self.z_dim + self.formula_dim + 1
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.LayerNorm(self.condition_dim),
        )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        *,
        device: Optional[torch.device] = None,
        adapter: str = "graph2mol",
    ) -> torch.Tensor:
        """Use the frozen parent formula policy and return its count vector.

        The parent parser rejects duplicate elements, leading-zero counts,
        unsupported adapter elements, H-only formulas and atom-budget
        violations.  Graph2Mol and MS2Mol share this formula policy; ``adapter='msg'`` explicitly
        rejects As/Se because the frozen MSG adapter does not support them.
        """
        from utils import formula as parser_module
        parsed = parser_module.parse_formula(
            str(formula), adapter=str(adapter), formula_kind="neutral", max_atoms=150,
        )
        values = {symbol: 0.0 for symbol in ELEMENT_ORDER}
        for symbol, count in parsed.counts:
            if symbol not in values:
                raise ValueError(f"Parent formula parser returned unsupported element: {symbol}")
            values[symbol] = float(count)
        return torch.tensor([values[symbol] for symbol in ELEMENT_ORDER], dtype=torch.float32, device=device)

    @classmethod
    def coerce_counts(
        cls,
        value: Union[torch.Tensor, Mapping[str, float], str, Sequence[float]],
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        adapter: str = "graph2mol",
    ) -> torch.Tensor:
        if isinstance(value, str):
            result = cls.from_formula(value, device=device, adapter=adapter)
        elif isinstance(value, Mapping):
            unknown = set(value) - set(ELEMENT_ORDER)
            if unknown:
                raise ValueError("Formula contains unsupported element keys: %s" % sorted(unknown))
            result = torch.tensor(
                [float(value.get(symbol, 0.0)) for symbol in ELEMENT_ORDER],
                device=device,
                dtype=dtype,
            )
        else:
            result = torch.as_tensor(value, device=device, dtype=dtype)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or result.shape[-1] != FORMULA_DIM:
            raise ValueError("formula_counts must have shape [batch, 14] in H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I order")
        if not torch.isfinite(result).all() or (result < 0).any():
            raise ValueError("formula_counts must be finite and non-negative")
        # Counts are part of the input contract.  Non-integral or negative
        # values are usually a silently corrupted formula and are rejected.
        if not torch.equal(result, result.round()):
            raise ValueError("formula_counts must contain integer neutral-element counts")
        return result

    def forward(self, z: torch.Tensor, formula_counts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.z_dim:
            raise ValueError("cond_emb_cached must have shape [batch, 512]")
        counts = self.coerce_counts(formula_counts, device=z.device, dtype=z.dtype)
        if counts.shape[0] != z.shape[0]:
            raise ValueError("formula_counts batch does not match cond_emb_cached")
        time = torch.as_tensor(t, device=z.device, dtype=z.dtype).reshape(-1, 1)
        if time.shape[0] != z.shape[0] or not torch.isfinite(time).all():
            raise ValueError("t must have one finite value per graph")
        # log1p makes the fixed count vector numerically well behaved while
        # preserving every element, including H, as an explicit channel.
        raw = torch.cat([z, torch.log1p(counts), time], dim=-1)
        return self.proj(raw)

def _zero_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

class SharedLayerCondition(nn.Module):
    """Per-layer node/edge FiLM residuals driven by one shared condition."""

    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.node_gamma = nn.Linear(condition_dim, hidden_dim)
        self.node_beta = nn.Linear(condition_dim, hidden_dim)
        self.edge_gamma = nn.Linear(condition_dim, hidden_dim)
        self.edge_beta = nn.Linear(condition_dim, hidden_dim)
        for layer in (self.node_gamma, self.node_beta, self.edge_gamma, self.edge_beta):
            _zero_linear(layer)

    def apply_node(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # This is a residual FiLM: at initialization gamma=beta=0 -> x exactly.
        return (1.0 + self.node_gamma(condition)) * x + self.node_beta(condition)

    def apply_edge(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return (1.0 + self.edge_gamma(condition)) * x + self.edge_beta(condition)

def _make_eq_cond_class(parent_cls):
    cached = _CLASS_CACHE.get(parent_cls)
    if cached is not None:
        return cached

    class _EQCondMS2Forge(parent_cls):
        """Parent MS2Forge plus isolated zero-initialized EQ-COND modulation."""

        def __init__(self, config, num_node_types, num_edge_types, atomic_numbers=None, eq_cond_dim=256):
            super().__init__(config, num_node_types, num_edge_types, atomic_numbers=atomic_numbers)
            if int(getattr(self.gnn, "cond_dim", 0)) != 0:
                raise RuntimeError("EQ-COND must leave the parent cond_dim=0 path disabled")
            if getattr(self.gnn, "triangle_updates", None) is not None:
                raise RuntimeError("EQ-COND cannot activate the parent triangle path")
            self.eq_cond_dim = int(eq_cond_dim)
            self.eq_cond_variant = EQ_COND_VARIANT
            self.parent_architecture_id = PARENT_ARCHITECTURE_ID
            self.parent_input_protocol_id = PARENT_INPUT_PROTOCOL_ID
            self.formula_condition = FormulaCondition(
                z_dim=self.contrastive_dim,
                condition_dim=self.eq_cond_dim,
            )
            self.layer_conditions = nn.ModuleList([
                SharedLayerCondition(self.eq_cond_dim, self.hidden_dim)
                for _ in range(self.num_layers)
            ])
            self.initialization = "EQ_COND_PARENT_PLUS_ZERO_FILM"
            self.is_eq_cond_architecture = True
            self.teacher_maturity = "PENDING_RESEARCH_BRANCH_NOT_FORMAL_TEACHER"


        def _assert_zero_initialized_condition(self):
            for layer in self.layer_conditions:
                for module in (layer.node_gamma, layer.node_beta, layer.edge_gamma, layer.edge_beta):
                    if module.weight.detach().abs().max().item() != 0.0:
                        raise RuntimeError("EQ-COND condition head is not zero initialized")
                    if module.bias is not None and module.bias.detach().abs().max().item() != 0.0:
                        raise RuntimeError("EQ-COND condition head bias is not zero initialized")

        def _run_eq_cond_gnn(self, node_features, edge_features, edge_index,
                             batch_node, batch_edge, condition_context):
            gnn = self.gnn
            layer_h_nodes = [] if getattr(self, "capture_layer_h_nodes", False) else None
            layer_h_edges = [] if getattr(self, "capture_layer_h_nodes", False) else None
            for i in range(gnn.num_blocks):
                node_update = gnn.node_blocks[i](node_features, edge_index, edge_features, cond=None)
                node_features = node_features + node_update
                edge_update = gnn.edge_blocks[i](edge_features, edge_index, node_features, cond=None)
                edge_features = edge_features + edge_update

                node_condition = condition_context.index_select(0, batch_node)
                edge_condition = condition_context.index_select(0, batch_edge)
                node_features = self.layer_conditions[i].apply_node(node_features, node_condition)
                edge_features = self.layer_conditions[i].apply_edge(edge_features, edge_condition)

                if i < gnn.num_blocks - 1:
                    node_features = F.dropout(node_features, p=gnn.dropout, training=self.training)
                    edge_features = F.dropout(edge_features, p=gnn.dropout, training=self.training)
                if layer_h_nodes is not None:
                    layer_h_nodes.append(node_features)
                    layer_h_edges.append(edge_features)
            if layer_h_nodes is not None:
                self._last_layer_h_nodes = layer_h_nodes
                self._last_layer_h_edges = layer_h_edges
            else:
                self._last_layer_h_nodes = None
                self._last_layer_h_edges = None
            return node_features, edge_features

        def forward(self, node_types=None, edge_index=None, batch_node=None, batch_edge=None,
                    instrument_type_idx=None, ionization_type_idx=None, t=None,
                    edge_types_t=None, pretrained_embedding=None, edge_theta=None,
                    spec_sos=None, spec_formula_array=None, spec_mask=None,
                    dense_X=None, dense_E=None, dense_y=None, dense_node_mask=None,
                    cond_emb_cached=None, node_noise=None, formula_counts=None):
            if cond_emb_cached is None:
                raise ValueError("EQ-COND requires Zmol/Zms cond_emb_cached")
            if formula_counts is None:
                raise ValueError("EQ-COND requires neutral formula_counts including H")
            if node_types is None or edge_index is None or batch_node is None or batch_edge is None:
                raise ValueError("EQ-COND sparse BFN inputs are required")
            cond_emb = cond_emb_cached.to(
                device=node_types.device,
                dtype=self.node_embedder.weight.dtype,
            )
            self._last_zms_raw = cond_emb
            if getattr(self, "use_zms_adapter", False) and self.zms_adapter is not None:
                cond_emb = self.zms_adapter(cond_emb)
            self._last_adapter_out = cond_emb
            return self._forward_eq_cond_bfn(
                node_types, edge_index, batch_node, batch_edge,
                instrument_type_idx, ionization_type_idx, t, edge_types_t,
                edge_theta, cond_emb, node_noise, formula_counts,
            )

        def _forward_eq_cond_bfn(self, node_types, edge_index, batch_node, batch_edge,
                                 instrument_type_idx, ionization_type_idx, t,
                                 edge_types_t, edge_theta, cond_emb, node_noise,
                                 formula_counts):
            if instrument_type_idx is None or ionization_type_idx is None:
                raise ValueError("BFN forward requires instrument_type_idx / ionization_type_idx")
            if t is None:
                raise ValueError("BFN forward requires t")
            dtype = self.node_embedder.weight.dtype
            condition_features = self.condition_embedding(instrument_type_idx, ionization_type_idx)
            time_features = self.time_embedding(t.to(dtype).unsqueeze(-1))
            node_onehot = F.one_hot(node_types, self.num_node_types).to(dtype)
            node_emb = self.node_embedder(node_onehot)
            scale = self.noise_scale if self.training else self.noise_scale_inference
            if scale:
                noise = torch.randn_like(node_emb) if node_noise is None else node_noise.to(node_emb)
                if noise.shape != node_emb.shape:
                    raise ValueError("Node noise shape mismatch")
                node_emb = node_emb + noise * scale
            node_cond = cond_emb.index_select(0, batch_node)
            node_condition = condition_features.index_select(0, batch_node)
            node_time = time_features.index_select(0, batch_node)
            node_features = self.node_proj(torch.cat([node_emb, node_cond, node_condition, node_time], dim=-1))
            edge_features = self.edge_init(node_features, edge_index)
            if edge_theta is not None:
                edge_state_emb = self.edge_theta_proj(edge_theta)
            else:
                edge_state_emb = self.edge_state_embedder(edge_types_t)
            edge_features = self.edge_state_fusion(torch.cat([edge_features, edge_state_emb], dim=-1))

            counts = FormulaCondition.coerce_counts(
                formula_counts, device=node_features.device, dtype=dtype,
            )
            if counts.shape[0] <= int(batch_node.max().item()):
                raise ValueError("formula_counts has fewer rows than batch_node")
            # The tower receives one time per graph.  Per-layer modulation uses
            # the same graph condition at nodes and edges, preserving endpoint
            # sharing and permutation equivariance.
            context = self.formula_condition(cond_emb, counts, t)
            node_features, edge_features = self._run_eq_cond_gnn(
                node_features, edge_features, edge_index, batch_node, batch_edge, context,
            )
            x0_logits = self.x0_predictor(edge_features)
            self._last_x0_logits = x0_logits
            return F.softmax(x0_logits, dim=-1)

        @torch.no_grad()
        def sample_bfn(self, node_types, edge_index, batch_node, batch_edge,
                       instrument_type_idx, ionization_type_idx,
                       pretrained_embedding=None, n_timesteps=None,
                       return_trajectory=False, disable_tqdm=False,
                       cond_emb_cached=None, formula_counts=None,
                       spec_sos=None, spec_formula_array=None, spec_mask=None,
                       noise_provider=None, noise_step_base=0,
                       return_probabilities=False):
            if cond_emb_cached is None or formula_counts is None:
                raise ValueError("EQ-COND sampling requires cond_emb_cached and formula_counts")
            if n_timesteps is None:
                n_timesteps = self.default_n_timesteps
            if int(n_timesteps) < 1:
                raise ValueError("BFN steps must be positive")
            device = node_types.device
            num_edges = edge_index.shape[1]
            batch_size = int(batch_node.max().item()) + 1
            k = self.num_edge_types
            beta1 = self.beta1.to(device)
            dtype = self.node_embedder.weight.dtype
            theta = torch.ones(num_edges, k, device=device, dtype=dtype) / k
            trajectory = [] if return_trajectory else None
            if return_trajectory:
                trajectory.append({"step": 0, "t": 0.0, "edge_types": theta.argmax(-1).cpu().numpy().copy(),
                                   "edge_probs": theta.detach().cpu().clone(), "theta": theta.detach().cpu().clone()})
            iterator = range(1, int(n_timesteps) + 1)
            if not disable_tqdm:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="  EQ-COND BFN 去噪", leave=False, position=2, mininterval=0.5)
            for i in iterator:
                t_val = (i - 1) / int(n_timesteps)
                t_batch = torch.full((batch_size,), max(t_val, 1e-4), device=device, dtype=dtype)
                node_noise = None if noise_provider is None else noise_provider.node(
                    noise_step_base + i, (len(node_types), self.node_dim), device, dtype)
                e_hat = self.forward(
                    node_types=node_types, edge_index=edge_index, batch_node=batch_node,
                    batch_edge=batch_edge, instrument_type_idx=instrument_type_idx,
                    ionization_type_idx=ionization_type_idx, t=t_batch,
                    edge_types_t=torch.zeros(num_edges, dtype=torch.long, device=device),
                    edge_theta=theta, node_noise=node_noise,
                    cond_emb_cached=cond_emb_cached, formula_counts=formula_counts,
                )
                alpha_i = beta1 * (2 * i - 1) / (int(n_timesteps) ** 2)
                mean = alpha_i * (k * e_hat - 1)
                std = (alpha_i * k).sqrt()
                eps = torch.randn_like(mean) if noise_provider is None else noise_provider.edge(
                    noise_step_base + i, mean.shape, device, dtype)
                # Form the noisy sender state before adding it to log(theta).
                # Regrouping these operations can change floating-point sampling.
                y = mean + eps * std
                theta = F.softmax(theta.log().clamp(min=-20) + y, dim=-1)
                if return_trajectory:
                    trajectory.append({"step": i, "t": i / int(n_timesteps),
                                       "edge_types": theta.argmax(-1).cpu().numpy().copy(),
                                       "edge_probs": theta.detach().cpu().clone(),
                                       "theta": theta.detach().cpu().clone(),
                                       "prediction_probs": e_hat.detach().cpu().clone()})
            t_final = torch.ones(batch_size, device=device, dtype=dtype)
            final = self.forward(
                node_types=node_types, edge_index=edge_index, batch_node=batch_node,
                batch_edge=batch_edge, instrument_type_idx=instrument_type_idx,
                ionization_type_idx=ionization_type_idx, t=t_final,
                edge_types_t=torch.zeros(num_edges, dtype=torch.long, device=device),
                edge_theta=theta,
                node_noise=None if noise_provider is None else noise_provider.node(
                    noise_step_base + int(n_timesteps) + 1, (len(node_types), self.node_dim), device, dtype),
                cond_emb_cached=cond_emb_cached, formula_counts=formula_counts,
            )
            labels = final.argmax(-1)
            self._last_sample_probabilities = final.detach()
            self._last_sample_theta = theta.detach()
            if return_trajectory:
                trajectory.append({"step": int(n_timesteps) + 1, "t": 1.0,
                                   "edge_types": labels.cpu().numpy().copy(),
                                   "edge_probs": final.detach().cpu().clone(),
                                   "theta": theta.detach().cpu().clone()})
                result = (labels, final, trajectory) if return_probabilities else (labels, trajectory)
                return result
            return (labels, final) if return_probabilities else labels

    _EQCondMS2Forge.__name__ = "EQCondMS2Forge"
    _EQCondMS2Forge.__qualname__ = "EQCondMS2Forge"
    _CLASS_CACHE[parent_cls] = _EQCondMS2Forge
    return _EQCondMS2Forge


class MS2Forge(_make_eq_cond_class(ParentMS2Forge)):
    """Public EQ-COND decoder; legacy directional checkpoints are rejected."""
    def __init__(self, config, num_node_types=14, num_edge_types=5,
                 atomic_numbers=None, eq_cond_dim=256):
        config = _namespace(normalize_model_config(config))
        if (config.contrastive_dim != 512 or config.gat.hidden_dim != 512
                or config.gat.num_layers != 6 or eq_cond_dim != 256):
            raise ValueError("EQ-COND requires z=512, h=512, six blocks, condition=256")
        if config.stage not in ("graph2mol", "ms2mol") or config.noise_scale != 0:
            raise ValueError("Unsupported EQ-COND stage or node noise scale")
        super().__init__(config, num_node_types, num_edge_types, atomic_numbers, eq_cond_dim)
        self.architecture_id = "eq_cond_unordered_bfn_v1"
        self.input_protocol_id = PARENT_INPUT_PROTOCOL_ID
        self.parameter_count = sum(p.numel() for p in self.parameters())
        self._assert_zero_initialized_condition()


def build_eq_cond_model(config, *, stage=None, device=None):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    payload = dict(config)
    payload = payload.get("model_config", payload)
    cfg = dict(payload.get("model", payload))
    cfg["stage"] = stage or cfg.get("stage", "graph2mol")
    atoms = payload.get("chem", {}).get("atomic_numbers",
        [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53])
    model = MS2Forge(cfg, 14, 5, atomic_numbers=atoms)
    model.use_zms_adapter = cfg["stage"] == "ms2mol"
    return model.to(device) if device is not None else model


# Import/pickle compatibility only; internal code uses the canonical names.
FLASH = MS2Forge
ParentFLASH = ParentMS2Forge
