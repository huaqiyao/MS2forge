




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch_scatter import scatter_sum, scatter_softmax



NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    'silu': nn.SiLU()
}


class MLP(Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(Linear(hidden_dim, out_dim))
            else:
                layers.append(Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BondFFN(Module):


    def __init__(self, bond_dim, node_dim, inter_dim, use_gate=True, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim + node_dim, out_dim, 32)

    def forward(self, bond_feat_input, node_feat_input):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class NodeBlock(Module):







    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate=True, cond_dim=0):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        self.cond_dim = cond_dim

        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim + node_dim, hidden_dim, hidden_dim)

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)


        if cond_dim > 0:
            self.film_mul = Linear(cond_dim, hidden_dim)
            self.film_add = Linear(cond_dim, hidden_dim)

            nn.init.zeros_(self.film_mul.weight)
            nn.init.zeros_(self.film_mul.bias)
            nn.init.zeros_(self.film_add.weight)
            nn.init.zeros_(self.film_add.bias)

    def forward(self, x, edge_index, edge_attr, cond=None):









        N = x.size(0)
        row, col = edge_index  # row: source, col: target

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H)
        msg_j = self.msg_net(h_edge * h_node[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages (scatter to target nodes)
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg


        if self.cond_dim > 0 and cond is not None:
            gamma = self.film_mul(cond)  # (N, hidden_dim)
            beta = self.film_add(cond)   # (N, hidden_dim)
            out = (gamma + 1) * out + beta

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class EdgeBlock(Module):







    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True, cond_dim=0):
        super().__init__()
        self.use_gate = use_gate
        self.cond_dim = cond_dim
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim


        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)


        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()


        if cond_dim > 0:
            self.film_mul = Linear(cond_dim, edge_dim)
            self.film_add = Linear(cond_dim, edge_dim)
            nn.init.zeros_(self.film_mul.weight)
            nn.init.zeros_(self.film_mul.bias)
            nn.init.zeros_(self.film_add.weight)
            nn.init.zeros_(self.film_add.bias)

    def forward(self, h_bond, bond_index, h_node, cond=None):









        N = h_node.size(0)
        left_node, right_node = bond_index


        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node])
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node])
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]


        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )


        if self.cond_dim > 0 and cond is not None:
            gamma = self.film_mul(cond)
            beta = self.film_add(cond)
            h_bond = (gamma + 1) * h_bond + beta

        h_bond = self.layer_norm(h_bond)
        h_bond = self.out_transform(self.act(h_bond))
        return h_bond


class TriangleMultiplicativeUpdate(Module):











    def __init__(self, dim):
        super().__init__()
        bottleneck = max(dim // 4, 32)
        self.left_proj  = Linear(dim, bottleneck)
        self.right_proj = Linear(dim, bottleneck)
        self.left_gate  = Linear(dim, bottleneck)
        self.right_gate = Linear(dim, bottleneck)
        self.out_proj   = nn.Sequential(Linear(bottleneck, dim))
        self.out_gate   = Linear(dim, dim)
        self.norm       = nn.LayerNorm(bottleneck)

        nn.init.zeros_(self.out_proj[0].weight); nn.init.zeros_(self.out_proj[0].bias)
        nn.init.zeros_(self.out_gate.weight); nn.init.zeros_(self.out_gate.bias)

    def forward(self, h_edge, edge_index, batch_edge):
        src, dst = edge_index
        N = max(src.max().item(), dst.max().item()) + 1

        left  = self.left_proj(h_edge)  * torch.sigmoid(self.left_gate(h_edge))
        right = self.right_proj(h_edge) * torch.sigmoid(self.right_gate(h_edge))


        left_agg  = scatter_sum(left,  dst, dim=0, dim_size=N)
        right_agg = scatter_sum(right, src, dim=0, dim_size=N)


        triangle = left_agg[src] * right_agg[dst]


        gate = torch.sigmoid(self.out_gate(h_edge))
        return h_edge + gate * self.out_proj(self.norm(triangle))


class EdgeBiasedAttention(Module):






    def __init__(self, node_dim, edge_dim, num_heads=4):
        super().__init__()
        assert node_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = Linear(node_dim, node_dim)
        self.k = Linear(node_dim, node_dim)
        self.v = Linear(node_dim, node_dim)
        self.out = Linear(node_dim, node_dim)


        self.e_mul = Linear(edge_dim, node_dim)
        self.e_add = Linear(edge_dim, node_dim)
        nn.init.zeros_(self.e_mul.weight); nn.init.zeros_(self.e_mul.bias)
        nn.init.zeros_(self.e_add.weight); nn.init.zeros_(self.e_add.bias)


        self.e_update = Linear(node_dim, edge_dim)
        nn.init.zeros_(self.e_update.weight); nn.init.zeros_(self.e_update.bias)

    def forward(self, h_node, h_edge, edge_index):
        N = h_node.shape[0]
        src, dst = edge_index  # halfedge: src < dst

        Q = self.q(h_node); K = self.k(h_node); V = self.v(h_node)
        q_i, k_j = Q[src], K[dst]


        attn = (q_i * k_j) * self.scale  # [E, D]
        attn = torch.clamp(attn, min=-10.0, max=10.0)



        e_mul = torch.tanh(self.e_mul(h_edge))       # (-1, 1)
        e_add = self.e_add(h_edge).clamp(-5.0, 5.0)
        attn = attn * (e_mul + 1) + e_add
        attn = torch.clamp(attn, min=-10.0, max=10.0)


        h_edge_delta = self.e_update(attn)

        # scatter softmax per src-node, per head
        attn_h = attn.reshape(-1, self.num_heads, self.head_dim).mean(-1)  # [E, H]
        attn_w = scatter_softmax(attn_h, src, dim=0).unsqueeze(-1)  # [E, H, 1]

        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        v_j = V[dst].reshape(-1, self.num_heads, self.head_dim)
        msg = (attn_w * v_j).reshape(-1, Q.shape[-1])
        aggr = scatter_sum(msg, src, dim=0, dim_size=N)

        return self.out(aggr), h_edge_delta


class NodeEdgeNet2D(Module):










    def __init__(self, node_dim, edge_dim, num_blocks, hidden_dim=None, use_gate=True, dropout=0.1, cond_dim=0, use_triangle=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.use_gate = use_gate
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.use_triangle = use_triangle

        if hidden_dim is None:
            hidden_dim = node_dim


        self.node_blocks = ModuleList()

        self.edge_blocks = ModuleList()

        self.global_node_projs = ModuleList() if cond_dim > 0 else None
        self.global_edge_projs = ModuleList() if cond_dim > 0 else None

        self.triangle_updates = ModuleList() if (cond_dim > 0 or use_triangle) else None

        self.edge_attns = None

        for _ in range(num_blocks):
            self.node_blocks.append(NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                use_gate=use_gate,
                cond_dim=cond_dim,
            ))
            self.edge_blocks.append(EdgeBlock(
                edge_dim=edge_dim,
                node_dim=node_dim,
                hidden_dim=hidden_dim,
                use_gate=use_gate,
                cond_dim=cond_dim,
            ))
            if cond_dim > 0:

                gn_proj = Linear(node_dim, node_dim)
                nn.init.zeros_(gn_proj.weight)
                nn.init.zeros_(gn_proj.bias)
                self.global_node_projs.append(gn_proj)
                ge_proj = Linear(edge_dim, edge_dim)
                nn.init.zeros_(ge_proj.weight)
                nn.init.zeros_(ge_proj.bias)
                self.global_edge_projs.append(ge_proj)
            if cond_dim > 0 or use_triangle:

                self.triangle_updates.append(TriangleMultiplicativeUpdate(node_dim))

    def forward(self, h_node, h_edge, edge_index, cond_node=None, cond_edge=None,
                batch_node=None, batch_edge=None, return_intermediate=False):














        layer_h_node_list = [] if return_intermediate else None
        for i in range(self.num_blocks):

            h_node_update = self.node_blocks[i](h_node, edge_index, h_edge, cond=cond_node)
            h_node = h_node + h_node_update


            h_edge_update = self.edge_blocks[i](h_edge, edge_index, h_node, cond=cond_edge)
            h_edge = h_edge + h_edge_update


            if self.triangle_updates is not None:
                h_edge = self.triangle_updates[i](h_edge, edge_index, batch_edge)


            if self.cond_dim > 0 and batch_node is not None:
                batch_size = batch_node.max().item() + 1

                global_node = scatter_sum(h_node, batch_node, dim=0, dim_size=batch_size)
                node_counts = scatter_sum(torch.ones(h_node.shape[0], 1, device=h_node.device),
                                          batch_node, dim=0, dim_size=batch_size).clamp(min=1)
                global_node = global_node / node_counts
                h_node = h_node + self.global_node_projs[i](global_node[batch_node])


                if batch_edge is not None:
                    global_edge = scatter_sum(h_edge, batch_edge, dim=0, dim_size=batch_size)
                    edge_counts = scatter_sum(torch.ones(h_edge.shape[0], 1, device=h_edge.device),
                                              batch_edge, dim=0, dim_size=batch_size).clamp(min=1)
                    global_edge = global_edge / edge_counts
                    h_edge = h_edge + self.global_edge_projs[i](global_edge[batch_edge])


            if i < self.num_blocks - 1:
                h_node = F.dropout(h_node, p=self.dropout, training=self.training)
                h_edge = F.dropout(h_edge, p=self.dropout, training=self.training)

            if return_intermediate:
                layer_h_node_list.append(h_node)

        if return_intermediate:
            return h_node, h_edge, layer_h_node_list
        return h_node, h_edge


class EdgePredictor(Module):





    def __init__(self, edge_dim, num_edge_types, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = edge_dim

        self.predictor = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, num_edge_types)
        )

    def forward(self, h_edge):
        """
        Args:
            h_edge: Edge features, (E, edge_dim)
        Returns:
            edge_logits: (E, num_edge_types)
        """
        return self.predictor(h_edge)


class ValenceAwareEdgePredictor(Module):













    def __init__(self, edge_dim, num_edge_types, hidden_dim=None, dropout=0.1,
                 bond_order_values=None, max_valence_table=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = edge_dim

        self.num_edge_types = num_edge_types


        self.base_predictor = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, num_edge_types)
        )



        self.valence_encoder = nn.Sequential(
            Linear(2, 32),
            nn.ReLU(),
            Linear(32, num_edge_types)
        )

        nn.init.zeros_(self.valence_encoder[-1].weight)
        nn.init.zeros_(self.valence_encoder[-1].bias)


        if bond_order_values is not None:
            self.register_buffer('bond_order_values', bond_order_values)
        if max_valence_table is not None:
            self.register_buffer('max_valence_table', max_valence_table)

        print(f"[INFO] Edge predictor: valence aware(zero-initialized modulation)")

    def forward(self, h_edge, node_types=None, edge_index=None, edge_types_t=None, mask_idx=None):











        base_logits = self.base_predictor(h_edge)


        if (node_types is None or edge_index is None or edge_types_t is None
                or not hasattr(self, 'bond_order_values')):
            return base_logits


        device = h_edge.device
        E = edge_types_t.shape[0]
        N = node_types.shape[0]
        src, dst = edge_index


        known_mask = (edge_types_t != mask_idx)
        edge_bo = torch.zeros(E, device=device)
        edge_bo[known_mask] = self.bond_order_values[edge_types_t[known_mask]]


        from torch_scatter import scatter_sum
        used_valence = (
            scatter_sum(edge_bo, src, dim=0, dim_size=N) +
            scatter_sum(edge_bo, dst, dim=0, dim_size=N)
        )


        max_valence = self.max_valence_table[node_types]
        remaining_valence = torch.clamp(max_valence - used_valence, min=0.0)


        remaining_src = remaining_valence[src].unsqueeze(-1)  # [E, 1]
        remaining_dst = remaining_valence[dst].unsqueeze(-1)  # [E, 1]
        valence_context = torch.cat([remaining_src, remaining_dst], dim=-1)  # [E, 2]


        valence_modulation = self.valence_encoder(valence_context)  # [E, num_edge_types]


        return base_logits + valence_modulation


class TwoStageEdgePredictor(Module):











    def __init__(self, edge_dim, num_edge_types, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = edge_dim

        self.shared = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bond_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, 2)
        )
        self.type_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, num_edge_types - 1)
        )

    def forward(self, h_edge):
        shared = self.shared(h_edge)
        bond_logits = self.bond_head(shared)
        no_bond_logit = bond_logits[:, 0:1]
        bond_logit = bond_logits[:, 1:2]
        type_logits = self.type_head(shared)
        return torch.cat([no_bond_logit, bond_logit + type_logits], dim=-1)


class FullyConnectedEdgeInit(Module):







    def __init__(self, node_dim, edge_dim, hidden_dim=None,
                 use_chem_prior=False, num_atom_types=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = edge_dim

        self.edge_init = nn.Sequential(
            Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, edge_dim)
        )


        self.use_chem_prior = use_chem_prior
        if use_chem_prior and num_atom_types is not None:


            self.chem_prior_embed = nn.Embedding(
                num_atom_types * num_atom_types,
                edge_dim
            )
            nn.init.zeros_(self.chem_prior_embed.weight)
            self.num_atom_types = num_atom_types
            print(f"[INFO] Edge initialization: chemical prior enabled(atom-pair embedding, zero initialization)")

    def forward(self, h_node, edge_index, node_types=None):








        left_node, right_node = edge_index
        left_feat = h_node[left_node]
        right_feat = h_node[right_node]
        edge_input = torch.cat([left_feat, right_feat], dim=-1)
        h_edge = self.edge_init(edge_input)


        if self.use_chem_prior and node_types is not None:
            left_type = node_types[left_node]
            right_type = node_types[right_node]

            pair_idx = torch.min(left_type, right_type) * self.num_atom_types + torch.max(left_type, right_type)
            chem_prior = self.chem_prior_embed(pair_idx)
            h_edge = h_edge + chem_prior

        return h_edge


###############################################################################
# ========================  Flash Mode: MMDiT  ========================


###############################################################################

class AdaLNZero(Module):






    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 3)
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):








        gamma, beta, gate = self.proj(cond).chunk(3, dim=-1)
        out = self.norm(x) * (1 + gamma) + beta
        return out, gate


class MMDiTJointAttention(Module):










    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0


        self.mol_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.spec_qkv = nn.Linear(dim, dim * 3, bias=False)


        self.mol_out_proj = nn.Linear(dim, dim)
        self.spec_out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, mol_tokens, spec_tokens, mol_batch=None, edge_attn_bias=None):










        B_mol = mol_tokens.shape[0]
        B_spec = spec_tokens.shape[0]


        mol_qkv = self.mol_qkv(mol_tokens).reshape(B_mol, 3, self.num_heads, self.head_dim)
        mol_q, mol_k, mol_v = mol_qkv[:, 0], mol_qkv[:, 1], mol_qkv[:, 2]


        spec_qkv = self.spec_qkv(spec_tokens).reshape(B_spec, 3, self.num_heads, self.head_dim)
        spec_q, spec_k, spec_v = spec_qkv[:, 0], spec_qkv[:, 1], spec_qkv[:, 2]





        if mol_batch is not None:

            batch_size = mol_batch.max().item() + 1
            num_spec_per_mol = B_spec // batch_size


            spec_k_grouped = spec_k.reshape(batch_size, num_spec_per_mol, self.num_heads, self.head_dim)
            spec_v_grouped = spec_v.reshape(batch_size, num_spec_per_mol, self.num_heads, self.head_dim)


            # mol_q: [N_mol, heads, head_dim]
            # spec_k for each node: [K, heads, head_dim]
            mol_spec_k = spec_k_grouped[mol_batch]  # [N_mol, K, heads, head_dim]
            mol_spec_v = spec_v_grouped[mol_batch]  # [N_mol, K, heads, head_dim]


            attn_mol2spec = torch.einsum('nhd,nkhd->nhk', mol_q, mol_spec_k) * self.scale
            attn_mol2spec = F.softmax(attn_mol2spec, dim=-1)
            attn_mol2spec = self.dropout(attn_mol2spec)


            mol_cross_out = torch.einsum('nhk,nkhd->nhd', attn_mol2spec, mol_spec_v)
            mol_cross_out = mol_cross_out.reshape(B_mol, self.dim)
            mol_out = self.mol_out_proj(mol_cross_out)




            mol_k_sum = scatter_sum(mol_k, mol_batch, dim=0, dim_size=batch_size)  # [B, heads, head_dim]
            mol_v_sum = scatter_sum(mol_v, mol_batch, dim=0, dim_size=batch_size)


            mol_k_for_spec = mol_k_sum.unsqueeze(1).expand(-1, num_spec_per_mol, -1, -1)  # [B, K, heads, hd]
            mol_v_for_spec = mol_v_sum.unsqueeze(1).expand(-1, num_spec_per_mol, -1, -1)

            mol_k_for_spec = mol_k_for_spec.reshape(B_spec, self.num_heads, self.head_dim)
            mol_v_for_spec = mol_v_for_spec.reshape(B_spec, self.num_heads, self.head_dim)


            attn_spec2mol = torch.einsum('nhd,nhd->nh', spec_q, mol_k_for_spec) * self.scale
            attn_spec2mol = torch.sigmoid(attn_spec2mol)


            spec_cross_out = (attn_spec2mol.unsqueeze(-1) * mol_v_for_spec).reshape(B_spec, self.dim)
            spec_out = self.spec_out_proj(spec_cross_out)
        else:

            mol_out = self.mol_out_proj(mol_tokens)
            spec_out = self.spec_out_proj(spec_tokens)

        return mol_out, spec_out


class MMDiTBlock(Module):













    def __init__(self, dim, cond_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.mol_adaln = AdaLNZero(dim, cond_dim)
        self.mol_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.mol_ffn_adaln = AdaLNZero(dim, cond_dim)


        self.spec_adaln = AdaLNZero(dim, cond_dim)
        self.spec_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.spec_ffn_adaln = AdaLNZero(dim, cond_dim)


        self.joint_attn = MMDiTJointAttention(dim, num_heads, dropout)

    def forward(self, mol_tokens, spec_tokens, cond_mol, cond_spec, mol_batch=None):












        mol_normed, mol_gate_attn = self.mol_adaln(mol_tokens, cond_mol)
        spec_normed, spec_gate_attn = self.spec_adaln(spec_tokens, cond_spec)

        mol_attn_out, spec_attn_out = self.joint_attn(mol_normed, spec_normed, mol_batch)

        mol_tokens = mol_tokens + mol_gate_attn * mol_attn_out
        spec_tokens = spec_tokens + spec_gate_attn * spec_attn_out

        # --- FFN ---
        mol_normed2, mol_gate_ffn = self.mol_ffn_adaln(mol_tokens, cond_mol)
        spec_normed2, spec_gate_ffn = self.spec_ffn_adaln(spec_tokens, cond_spec)

        mol_tokens = mol_tokens + mol_gate_ffn * self.mol_ffn(mol_normed2)
        spec_tokens = spec_tokens + spec_gate_ffn * self.spec_ffn(spec_normed2)

        return mol_tokens, spec_tokens


class SpectrumGraphMMDiT(Module):







    def __init__(self, node_dim, edge_dim, cond_dim, num_mmdit_blocks=2,
                 num_heads=8, num_spec_tokens=16, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_spec_tokens = num_spec_tokens


        self.spec_tokenizer = nn.Sequential(
            nn.Linear(1024, num_spec_tokens * node_dim),
            nn.GELU(),
        )

        # MMDiT blocks
        self.mmdit_blocks = ModuleList([
            MMDiTBlock(node_dim, cond_dim, num_heads, dropout)
            for _ in range(num_mmdit_blocks)
        ])


        self.spec_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim)
        )


        self.output_gate = nn.Parameter(torch.zeros(node_dim))

    def forward(self, node_features, pretrained_embedding, cond_features, batch_node):









        batch_size = pretrained_embedding.shape[0]


        spec_tokens = self.spec_tokenizer(pretrained_embedding).reshape(
            batch_size, self.num_spec_tokens, self.node_dim
        )

        spec_tokens_flat = spec_tokens.reshape(batch_size * self.num_spec_tokens, self.node_dim)



        cond_dim = cond_features.shape[-1]

        cond_per_mol = scatter_sum(cond_features, batch_node, dim=0, dim_size=batch_size)

        nodes_per_mol = scatter_sum(torch.ones(batch_node.shape[0], 1, device=batch_node.device),
                                     batch_node, dim=0, dim_size=batch_size)
        cond_per_mol = cond_per_mol / nodes_per_mol.clamp(min=1)

        cond_spec = cond_per_mol.unsqueeze(1).expand(-1, self.num_spec_tokens, -1).reshape(
            batch_size * self.num_spec_tokens, cond_dim
        )


        for block in self.mmdit_blocks:
            node_features, spec_tokens_flat = block(
                node_features, spec_tokens_flat,
                cond_mol=cond_features,
                cond_spec=cond_spec,
                mol_batch=batch_node
            )


        spec_pooled = spec_tokens_flat.reshape(batch_size, self.num_spec_tokens, self.node_dim)
        spec_global = self.spec_pool(spec_pooled.mean(dim=1))  # [B, node_dim]

        node_features = node_features + self.output_gate * spec_global[batch_node]

        return node_features


###############################################################################
# ==================  Flash Mode: FlowTrans Denoiser  ==================


###############################################################################

class OuterProductMean(Module):






    def __init__(self, node_dim, edge_dim, c_hidden=32):
        super().__init__()
        self.c_hidden = c_hidden
        self.proj_a = Linear(node_dim, c_hidden)
        self.proj_b = Linear(node_dim, c_hidden)
        self.out_proj = Linear(c_hidden * c_hidden, edge_dim)

    def forward(self, node_features, edge_index):
        src, dst = edge_index
        a = self.proj_a(node_features[src])
        b = self.proj_b(node_features[dst])
        outer = torch.einsum('ec,ed->ecd', a, b).reshape(a.shape[0], -1)
        return self.out_proj(outer)


class DiGressStyleAttention(Module):





    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        self.edge_mul = Linear(d_model, d_model)
        self.edge_add = Linear(d_model, d_model)
        nn.init.zeros_(self.edge_mul.weight); nn.init.zeros_(self.edge_mul.bias)
        nn.init.zeros_(self.edge_add.weight); nn.init.zeros_(self.edge_add.bias)

        self.edge_out_proj = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_features, edge_index):
        N = node_features.shape[0]
        H, D = self.num_heads, self.head_dim
        src, dst = edge_index

        Q = self.q_proj(node_features)
        K = self.k_proj(node_features)
        V = self.v_proj(node_features)

        q_i, k_j = Q[src], K[dst]
        attn = (q_i * k_j) * self.scale
        attn = torch.clamp(attn, min=-10.0, max=10.0)

        e_mul = torch.tanh(self.edge_mul(edge_features))
        e_add = self.edge_add(edge_features).clamp(-5.0, 5.0)
        attn = attn * (e_mul + 1) + e_add
        attn = torch.clamp(attn, min=-10.0, max=10.0)

        new_edge_features = self.edge_out_proj(attn)

        attn_h = attn.reshape(-1, H, D).mean(-1)
        attn_w = scatter_softmax(attn_h, src, dim=0).unsqueeze(-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.dropout(attn_w)

        v_j = V[dst].reshape(-1, H, D)
        msg = (attn_w * v_j).reshape(-1, self.d_model)
        aggr = scatter_sum(msg, src, dim=0, dim_size=N)

        return self.out_proj(aggr), new_edge_features


class FlowTransTriangleUpdate(Module):





    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(dim // 4, 32)
        self.norm = nn.LayerNorm(dim)
        self.left_proj = Linear(dim, hidden_dim)
        self.right_proj = Linear(dim, hidden_dim)
        self.left_gate = Linear(dim, hidden_dim)
        self.right_gate = Linear(dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = Linear(hidden_dim, dim)
        self.out_gate = Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_gate.weight); nn.init.zeros_(self.out_gate.bias)

    def forward(self, h_edge, edge_index):
        src, dst = edge_index
        N = max(src.max().item(), dst.max().item()) + 1
        h = self.norm(h_edge)
        left = self.left_proj(h) * torch.sigmoid(self.left_gate(h))
        right = self.right_proj(h) * torch.sigmoid(self.right_gate(h))
        left_agg = scatter_sum(left, dst, dim=0, dim_size=N)
        right_agg = scatter_sum(right, src, dim=0, dim_size=N)
        triangle = left_agg[src] * right_agg[dst]
        gate = torch.sigmoid(self.out_gate(h_edge))
        return h_edge + gate * self.out_proj(self.out_norm(triangle))


class GatedEdgeUpdate(Module):

    def __init__(self, edge_dim, node_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = edge_dim

        self.bond_linear_left = Linear(edge_dim, hidden_dim, bias=False)
        self.node_linear_left = Linear(node_dim, hidden_dim, bias=False)
        self.gate_left = nn.Sequential(
            Linear(edge_dim + node_dim, 32), nn.LayerNorm(32), nn.ReLU(), Linear(32, hidden_dim))

        self.bond_linear_right = Linear(edge_dim, hidden_dim, bias=False)
        self.node_linear_right = Linear(node_dim, hidden_dim, bias=False)
        self.gate_right = nn.Sequential(
            Linear(edge_dim + node_dim, 32), nn.LayerNorm(32), nn.ReLU(), Linear(32, hidden_dim))

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)
        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(edge_dim, edge_dim)

    def forward(self, h_edge, edge_index, h_node):
        N = h_node.size(0)
        left_node, right_node = edge_index

        bl = self.bond_linear_left(h_edge) * self.node_linear_left(h_node[left_node])
        bl = bl * torch.sigmoid(self.gate_left(torch.cat([h_edge, h_node[left_node]], -1)))
        msg_left = scatter_sum(bl, right_node, dim=0, dim_size=N)[left_node]

        br = self.bond_linear_right(h_edge) * self.node_linear_right(h_node[right_node])
        br = br * torch.sigmoid(self.gate_right(torch.cat([h_edge, h_node[right_node]], -1)))
        msg_right = scatter_sum(br, left_node, dim=0, dim_size=N)[right_node]

        h_new = (msg_left + msg_right
                 + self.node_ffn_left(h_node[left_node])
                 + self.node_ffn_right(h_node[right_node])
                 + self.self_ffn(h_edge))
        return self.out_transform(self.act(self.layer_norm(h_new)))


class FlowTransBlock(Module):









    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.node_norm = nn.LayerNorm(d_model)
        self.attention = DiGressStyleAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            Linear(d_ff, d_model), nn.Dropout(dropout))

        self.edge_update = GatedEdgeUpdate(d_model, d_model, d_model, dropout)

        self.edge_ffn_norm = nn.LayerNorm(d_model)
        self.edge_ffn = nn.Sequential(
            Linear(d_model, d_ff // 2), nn.GELU(), nn.Dropout(dropout),
            Linear(d_ff // 2, d_model), nn.Dropout(dropout))

        self.triangle_update = FlowTransTriangleUpdate(d_model)

    def forward(self, node_features, edge_features, edge_index):

        node_delta, new_edge = self.attention(self.node_norm(node_features), edge_features, edge_index)
        node_features = node_features + self.dropout(node_delta)
        edge_features = edge_features + new_edge


        node_features = node_features + self.ffn(self.ffn_norm(node_features))


        edge_features = edge_features + self.edge_update(edge_features, edge_index, node_features)

        # 4. Edge FFN
        edge_features = edge_features + self.edge_ffn(self.edge_ffn_norm(edge_features))

        # 5. Triangle Update
        edge_features = self.triangle_update(edge_features, edge_index)

        return node_features, edge_features


class FlowTransDenoiser(Module):





    def __init__(self, node_dim, edge_dim, num_layers, num_heads, d_ff,
                 num_edge_classes, dropout=0.1, opm_hidden=32):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.edge_state_embedder = nn.Embedding(num_edge_classes, edge_dim)
        self.opm = OuterProductMean(node_dim, edge_dim, c_hidden=opm_hidden)
        self.edge_fusion = Linear(edge_dim * 2, edge_dim)

        self.blocks = ModuleList([
            FlowTransBlock(node_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.edge_predictor = nn.Sequential(
            Linear(edge_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout),
            Linear(edge_dim, edge_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            Linear(edge_dim // 2, num_edge_classes - 1))

    def forward(self, node_features, edge_index, edge_types_t):
        edge_init = self.opm(node_features, edge_index)
        edge_state = self.edge_state_embedder(edge_types_t)
        edge_features = self.edge_fusion(torch.cat([edge_init, edge_state], dim=-1))

        for block in self.blocks:
            node_features, edge_features = block(node_features, edge_features, edge_index)

        return self.edge_predictor(edge_features)


def get_complete_graph(batch):









    device = batch.device


    natoms = scatter_sum(torch.ones_like(batch), index=batch, dim=0)


    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)


    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)


    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])


    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms

    return edge_index, num_edges

