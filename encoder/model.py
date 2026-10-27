"""Dual encoder with tensor-only scattering and a global contrastive loss."""
from types import SimpleNamespace
import torch
from torch import nn
from .reference import Contrastive_model
from .finetune_math import compute_contrastive_loss


def build(temperature=15.0):
    return Contrastive_model(hidden_dim=512, num_transformer_layers=3, nhead=8,
                             embeddings_dim=512, dropout=0.1, input_dropout=0.1,
                             max_len=129, graph=True, fp_pred=False, fp_length=2048,
                             initial_temperature=temperature, trainable_temperature=True)


class TensorForward(nn.Module):
    """Scatter inputs only; gather embeddings before forming the full loss."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sos, formula, mask, X, E, y, node_mask):
        graph = SimpleNamespace(X=X, E=E, y=y, node_mask=node_mask)
        return self.model(sos, formula, graph=graph, mask=mask)


def finetune_loss(model, embeddings):
    return compute_contrastive_loss(model, *embeddings)


def load_weights(model, filename):
    obj = torch.load(filename, map_location='cpu', weights_only=True)
    state = obj.get('model', obj.get('state_dict', obj)) if isinstance(obj, dict) else obj
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict or a model/state_dict wrapper')
    model.load_state_dict(state, strict=True)
