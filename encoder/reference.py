"""Spectrum and molecular graph encoder architectures for contrastive learning.

Defines the spectrum Transformer, graph Transformer, and dual-encoder model
used for pretraining and fine-tuning.
"""
import math
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

class PlaceHolder:
    def __init__(self, X, E, y, atom_attr=None, smiles = None):
        self.X = X
        self.E = E
        self.y = y
        self.atom_attr = atom_attr
        self.smiles = smiles

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if self.atom_attr != None:
            self.atom_attr = self.atom_attr.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        
        if self.atom_attr != None:
            if collapse:
                self.charges = torch.argmax(self.atom_attr[:, :, :9], dim=-1)
                self.charges[node_mask == 0] = - 1
                self.Hs = torch.argmax(self.atom_attr[:, :, 9:], dim=-1)
                self.Hs[node_mask == 0] = - 1
            
            else:
                self.charges = self.atom_attr[:, :, :9] * x_mask
                self.Hs = self.atom_attr[:, :, 9:] * x_mask
        
        return self

utils = SimpleNamespace(PlaceHolder=PlaceHolder)
diffusion_utils = SimpleNamespace(assert_correctly_masked=assert_correctly_masked)

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y

class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y

class GraphTransformer_embedding(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        new_X = self.mlp_in_X(X)

        after_in = utils.PlaceHolder(X=new_X, E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Only sum over valid nodes according to node_mask
        mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
        X_masked = X * mask
        valid_counts = mask.sum(dim=1).clamp(min=1)  # (bs, 1)
        X_mean = X_masked.sum(dim=1) / valid_counts
        X = self.mlp_out_X(X_mean)

        return X

class TransformerModel(nn.Module):
    def __init__(self,
                 dim_sos=13,
                 dim_formula=144,
                 hidden_dim=256,
                 num_transformer_layers=4,
                 nhead=8,
                 output_dim=512,
                 dropout=0.1,
                 input_dropout = 0.1,
                 max_len=129):
        """
        Args:
            dim_sos (int): Input dimension for the sos token.
            dim_formula (int): Input dimension for each formula token.
            hidden_dim (int): The dimension to project tokens to.
            num_transformer_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate used in positional encoding and transformer.
        """
        super(TransformerModel, self).__init__()

        # Linear projections to go from input dimensions to hidden_dim.
        self.sos_proj = nn.Linear(dim_sos, hidden_dim)
        self.formula_proj = nn.Linear(dim_formula, hidden_dim)
        
        # Sinusoidal positional encoding.
        self.input_dropout = nn.Dropout(p=input_dropout)
        
        # Transformer encoder definition.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        
        # A 2-layer MLP to process the first token after the transformer.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2048)
        )
        
        # Final layer normalization.
        self.norm = nn.LayerNorm(hidden_dim)

        scale = hidden_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(hidden_dim, output_dim))

        
    def forward(self, sos, formula_array, mask=None, fp = True):
        """
        Args:
            sos: Tensor of shape (B, 1, 13) containing the first token.
            formula_array: Tensor of shape (B, 128, 144) containing the rest of the tokens.
            mask: Optional boolean Tensor of shape (B, 129) where True indicates positions to ignore.
                  This will be passed as src_key_padding_mask to the transformer.
        Returns:
            Tensor of shape (B, hidden_dim) after processing.
        """
        # Project the inputs to hidden_dim.
        sos_proj = self.sos_proj(sos)  # (B, 1, d)
        formula_proj = self.formula_proj(formula_array)  # (B, 128, d)
        
        # Concatenate along the token dimension: (B, 129, d)
        tokens = torch.cat([sos_proj, formula_proj], dim=1)
        
        # Transformer encoder expects shape (seq_len, batch, hidden_dim).
        tokens = tokens.transpose(0, 1)  # (129, B, d)
        
        # # Add sinusoidal positional encoding.
        tokens = self.input_dropout(tokens)
        
        # Pass through the transformer encoder.
        transformer_out = self.transformer_encoder(tokens, src_key_padding_mask=mask)  # (129, B, d),  (B, 129)
        
        # Transpose back to (B, 129, 512).
        transformer_out = transformer_out.transpose(0, 1)
        
        # Take the first token (sos) from each batch element.
        first_token = transformer_out[:, 0, :]  # (B, d)
        output = self.norm (first_token) @ self.proj
        
        return output

class Contrastive_model(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 max_len = 129,
                 num_transformer_layers=4,
                 nhead=8,
                 embeddings_dim=512,
                 dropout=0.1,
                 input_dropout = 0.1,
                 fp_length = 2048,
                 graph = False,
                 fp_pred = False,
                 initial_temperature = 30.0,
                 trainable_temperature = False):    
        """
        Args:
            dim_sos (int): Input dimension for the sos token.
            dim_formula (int): Input dimension for each formula token.
            hidden_dim (int): The dimension to project tokens to.
            num_transformer_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate used in positional encoding and transformer.
        """
        super(Contrastive_model, self).__init__()

        self.ms_encoder = TransformerModel(
                hidden_dim=hidden_dim,
                max_len = max_len,
                num_transformer_layers=num_transformer_layers,
                nhead=nhead,
                output_dim=embeddings_dim,
                dropout=dropout,
                input_dropout = input_dropout)

        self.graph = graph
        self.fp_pred = fp_pred

        if self.graph:
            self.graph_encoder = GraphTransformer_embedding(n_layers=4,
                                input_dims={'X': 11, 'E': 5, 'y':1},
                                hidden_mlp_dims={'X': 256, 'E': 128, 'y': 1},
                                hidden_dims={'dx': 256, 'de': 128, 'dy': 1, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1},
                                output_dims={'X': embeddings_dim},
                                act_fn_in=nn.ReLU(),
                                act_fn_out=nn.ReLU())

        if self.fp_pred:
            self.out_mlp = nn.Sequential(
            nn.Linear(embeddings_dim, embeddings_dim),
            nn.ReLU(),
            nn.Linear(embeddings_dim, fp_length))
        
        if trainable_temperature:
            self.inv_temperature = nn.Parameter(torch.tensor(1.0/initial_temperature))

    
    def forward(self, sos, formula_array, fp=None, graph = None, mask=None):

        ms_embeddings = self.ms_encoder(sos, formula_array, mask).float()

        if self.graph:
            graph_embeddings = self.graph_encoder(graph.X, graph.E, graph.y, graph.node_mask)

            if self.fp_pred:
                fp_output = self.out_mlp(ms_embeddings)
                return ms_embeddings, graph_embeddings, fp_output
            
            else:
                return ms_embeddings, graph_embeddings

        elif self.fp_pred:
            fp_output = self.out_mlp(ms_embeddings)
            return fp_output

        else:
            return ms_embeddings
