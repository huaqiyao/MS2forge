"""
高效的2D图神经网络模块
参考 FormulaDiff 的实现，使用 torch_scatter 进行向量化操作，避免 for 循环
"""


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Module, ModuleList, Linear


from torch_scatter import scatter_sum


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
    """边特征前馈网络：结合边特征和节点特征"""

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
    """
    节点更新模块
    使用 scatter_sum 进行消息聚合，避免 for 循环

    MS2Forge 模式：支持可选的 FiLM 条件调制（DiGress 风格）
    """

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

        # FiLM 条件调制（DiGress 风格：在 LayerNorm 之前用条件调制）
        if cond_dim > 0:
            self.film_mul = Linear(cond_dim, hidden_dim)
            self.film_add = Linear(cond_dim, hidden_dim)
            # 初始化为恒等：(0+1)*x + 0 = x
            nn.init.zeros_(self.film_mul.weight)
            nn.init.zeros_(self.film_mul.bias)
            nn.init.zeros_(self.film_add.weight)
            nn.init.zeros_(self.film_add.bias)

    def forward(self, x, edge_index, edge_attr, cond=None):
        """
        Args:
            x: Node features, (N, H)
            edge_index: (2, E)
            edge_attr: (E, H)
            cond: 可选，节点级条件特征 (N, cond_dim)，MS2Forge 模式使用
        Returns:
            Node feature updates, (N, H)
        """
        N = x.size(0)
        row, col = edge_index  # row: source, col: target

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H)
        # One state per unordered edge; both endpoints receive the same shared
        # message rule evaluated at their own neighbour's features.
        neighbour = torch.cat([h_node[col], h_node[row]], dim=0)
        edges_twice = torch.cat([h_edge, h_edge], dim=0)
        messages = self.msg_net(edges_twice * neighbour)
        if self.use_gate:
            gate = self.gate(torch.cat([
                torch.cat([edge_attr, edge_attr], dim=0),
                torch.cat([x[col], x[row]], dim=0),
            ], dim=-1))
            messages = messages * torch.sigmoid(gate)
        aggr_msg = scatter_sum(messages, torch.cat([row, col]), dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg

        # FiLM 条件调制（DiGress 风格）
        if self.cond_dim > 0 and cond is not None:
            gamma = self.film_mul(cond)  # (N, hidden_dim)
            beta = self.film_add(cond)   # (N, hidden_dim)
            out = (gamma + 1) * out + beta

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class EdgeBlock(Module):
    """
    边更新模块
    使用 scatter_sum 进行消息聚合，避免 for 循环

    MS2Forge 模式：支持可选的 FiLM 条件调制（DiGress 风格）
    """

    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True, cond_dim=0):
        super().__init__()
        self.use_gate = use_gate
        self.cond_dim = cond_dim
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        # 从左右两侧聚合邻居边的信息
        self.bond_ffn = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        # 从两端节点获取信息
        self.node_ffn = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

        # FiLM 条件调制（DiGress 风格）
        if cond_dim > 0:
            self.film_mul = Linear(cond_dim, edge_dim)
            self.film_add = Linear(cond_dim, edge_dim)
            nn.init.zeros_(self.film_mul.weight)
            nn.init.zeros_(self.film_mul.bias)
            nn.init.zeros_(self.film_add.weight)
            nn.init.zeros_(self.film_add.bias)

    def forward(self, h_bond, bond_index, h_node, cond=None):
        """
        Args:
            h_bond: Edge features, (E, edge_dim)
            bond_index: (2, E) - [left_node, right_node]
            h_node: Node features, (N, node_dim)
            cond: 可选，边级条件特征 (E, cond_dim)，MS2Forge 模式使用
        Returns:
            Edge feature updates, (E, edge_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # C_i = sum_{k != i} B(e_ik, h_k), over ALL incident edges.
        # Exclude this edge's two messages before adding the original self path.
        E = h_bond.size(0)
        messages = self.bond_ffn(
            torch.cat([h_bond, h_bond], dim=0),
            torch.cat([h_node[right_node], h_node[left_node]], dim=0),
        )
        to_left, to_right = messages[:E], messages[E:]
        context = scatter_sum(messages, torch.cat([left_node, right_node]), dim=0, dim_size=N)
        incident_other = (context[left_node] - to_left) + (context[right_node] - to_right)
        endpoint = self.node_ffn(h_node)
        h_bond = (
            incident_other
            + endpoint[left_node] + endpoint[right_node]
            + self.self_ffn(h_bond)
        )

        # FiLM 条件调制（DiGress 风格）
        if self.cond_dim > 0 and cond is not None:
            gamma = self.film_mul(cond)
            beta = self.film_add(cond)
            h_bond = (gamma + 1) * h_bond + beta

        h_bond = self.layer_norm(h_bond)
        h_bond = self.out_transform(self.act(h_bond))
        return h_bond


class NodeEdgeNet2D(Module):
    """
    2D 节点-边联合更新网络
    同时更新节点特征和边特征，适用于2D分子结构预测

    与 FormulaDiff 的 NodeEdgeNet 不同：
    - 去掉了 pos 相关的部分（不需要3D坐标）
    - 去掉了 distance_expansion（不需要距离特征）
    - 专注于节点和边特征的联合更新
    """

    def __init__(self, node_dim, edge_dim, num_blocks, hidden_dim=None, use_gate=True, dropout=0.1, cond_dim=0, use_triangle=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.use_gate = use_gate
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.use_triangle = use_triangle
        if cond_dim != 0 or use_triangle:
            raise ValueError("Use the EQ-COND layer conditioning in models.model; legacy GNN branches are not supported")

        if hidden_dim is None:
            hidden_dim = node_dim

        # 节点更新层
        self.node_blocks = ModuleList()
        # 边更新层
        self.edge_blocks = ModuleList()
        # 全局聚合层（MS2Forge 模式）
        self.global_node_projs = ModuleList() if cond_dim > 0 else None
        self.global_edge_projs = ModuleList() if cond_dim > 0 else None
        # Triangle Multiplicative Update（MS2Forge 模式，AlphaFold2 风格）
        self.triangle_updates = None
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
                # 全局图特征 → 节点/边（每层一个投影，零初始化）
                gn_proj = Linear(node_dim, node_dim)
                nn.init.zeros_(gn_proj.weight)
                nn.init.zeros_(gn_proj.bias)
                self.global_node_projs.append(gn_proj)
                ge_proj = Linear(edge_dim, edge_dim)
                nn.init.zeros_(ge_proj.weight)
                nn.init.zeros_(ge_proj.bias)
                self.global_edge_projs.append(ge_proj)

    def forward(self, h_node, h_edge, edge_index, cond_node=None, cond_edge=None,
                batch_node=None, batch_edge=None, return_intermediate=False):
        """
        Args:
            h_node: Node features, (N, node_dim)
            h_edge: Edge features, (E, edge_dim)
            edge_index: (2, E)
            cond_node: 可选，节点级条件 (N, cond_dim)，MS2Forge 模式使用
            cond_edge: 可选，边级条件 (E, cond_dim)，MS2Forge 模式使用
            batch_node: 可选，节点 batch 索引 (N,)，全局聚合用
            batch_edge: 可选，边 batch 索引 (E,)，全局聚合用
            return_intermediate: 若 True，额外返回 List[h_node_after_layer_i]（特征蒸馏用）
        Returns:
            return_intermediate=False: (h_node, h_edge)
            return_intermediate=True : (h_node, h_edge, layer_h_node_list)
        """
        layer_h_node_list = [] if return_intermediate else None
        self._last_layer_h_edges = [] if return_intermediate else None
        for i in range(self.num_blocks):
            # 节点更新
            h_node_update = self.node_blocks[i](h_node, edge_index, h_edge, cond=cond_node)
            h_node = h_node + h_node_update  # 残差连接

            # 边更新
            h_edge_update = self.edge_blocks[i](h_edge, edge_index, h_node, cond=cond_edge)
            h_edge = h_edge + h_edge_update  # 残差连接

            # 全局图特征聚合（MS2Forge 模式）
            if self.cond_dim > 0 and batch_node is not None:
                batch_size = batch_node.max().item() + 1
                # 节点全局均值 → 广播回每个节点
                global_node = scatter_sum(h_node, batch_node, dim=0, dim_size=batch_size)
                node_counts = scatter_sum(torch.ones(h_node.shape[0], 1, device=h_node.device),
                                          batch_node, dim=0, dim_size=batch_size).clamp(min=1)
                global_node = global_node / node_counts
                h_node = h_node + self.global_node_projs[i](global_node[batch_node])

                # 边全局均值 → 广播回每条边
                if batch_edge is not None:
                    global_edge = scatter_sum(h_edge, batch_edge, dim=0, dim_size=batch_size)
                    edge_counts = scatter_sum(torch.ones(h_edge.shape[0], 1, device=h_edge.device),
                                              batch_edge, dim=0, dim_size=batch_size).clamp(min=1)
                    global_edge = global_edge / edge_counts
                    h_edge = h_edge + self.global_edge_projs[i](global_edge[batch_edge])

            # Dropout (除了最后一层)
            if i < self.num_blocks - 1:
                h_node = F.dropout(h_node, p=self.dropout, training=self.training)
                h_edge = F.dropout(h_edge, p=self.dropout, training=self.training)

            if return_intermediate:
                layer_h_node_list.append(h_node)
                self._last_layer_h_edges.append(h_edge)

        if return_intermediate:
            return h_node, h_edge, layer_h_node_list
        return h_node, h_edge


class FullyConnectedEdgeInit(Module):
    """
    全连接图的边特征初始化
    从两端节点特征初始化边特征

    MS2Forge 模式：可选的化学先验增强
    """

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

        # MS2Forge 模式：化学先验边初始化
        self.use_chem_prior = use_chem_prior
        if use_chem_prior and num_atom_types is not None:
            # 原子对成键倾向嵌入（对称矩阵，零初始化）
            # 例如 C-C, C-N, C-O 等原子对的化学亲和性
            self.chem_prior_embed = nn.Embedding(
                num_atom_types * num_atom_types,
                edge_dim
            )
            nn.init.zeros_(self.chem_prior_embed.weight)
            self.num_atom_types = num_atom_types
            print(f"[INFO] 边初始化：启用化学先验（原子对嵌入，零初始化）")

    def forward(self, h_node, edge_index, node_types=None):
        """
        Args:
            h_node: Node features, (N, node_dim)
            edge_index: (2, E)
            node_types: (N,) 节点类型索引（MS2Forge 模式需要）
        Returns:
            h_edge: Initial edge features, (E, edge_dim)
        """
        left_node, right_node = edge_index
        left_feat = h_node[left_node]
        right_feat = h_node[right_node]
        # Shared ordered MLP, symmetrised without compressing endpoint vectors.
        edge_input = torch.cat([
            torch.cat([left_feat, right_feat], dim=-1),
            torch.cat([right_feat, left_feat], dim=-1),
        ], dim=0)
        both = self.edge_init(edge_input)
        E = left_feat.size(0)
        h_edge = 0.5 * (both[:E] + both[E:])

        # MS2Forge 模式：加入化学先验
        if self.use_chem_prior and node_types is not None:
            left_type = node_types[left_node]
            right_type = node_types[right_node]
            # 对称索引：min(a,b) * num_types + max(a,b)
            pair_idx = torch.min(left_type, right_type) * self.num_atom_types + torch.max(left_type, right_type)
            chem_prior = self.chem_prior_embed(pair_idx)
            h_edge = h_edge + chem_prior  # 零初始化，训练初期无影响

        return h_edge

