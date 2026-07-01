import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from torch_geometric.transforms import Compose  # imported by train.py

from utils.data import Drug3DData
from utils.dataset import *
from utils.misc import *


class FeaturizeMol2D(object):
    """
    2D分子特征化（无坐标）
    将简化的分子数据转换为模型输入格式
    """
    def __init__(self, atomic_numbers, mol_bond_types,
                 use_mask_node=False, use_mask_edge=False):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)

        self.num_node_types = self.num_element + int(use_mask_node)
        self.num_edge_types = self.num_bond_types + 1 + int(use_mask_edge)  # +1 for no-bond
        self.use_mask_node = use_mask_node
        self.use_mask_edge = use_mask_edge

        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}

        self.follow_batch = ['node_type', 'halfedge_type']
        self.exclude_keys = ['smiles', 'mol_id', 'edge_index', 'edge_type']

    def __call__(self, data):
        """
        输入数据格式（来自MSFileDataset）:
        - node_type: 原子序数 [num_atoms]
        - edge_index: 真实边索引 [2, num_edges*2]（双向）
        - edge_type: 真实边类型 [num_edges*2]（双向）
        """
        num_atoms = data.num_nodes

        # 将原子序数转换为节点类型索引
        node_type_indices = []
        for atomic_num in data.node_type.tolist():
            if atomic_num in self.ele_to_nodetype:
                node_type_indices.append(self.ele_to_nodetype[atomic_num])
            else:
                raise ValueError(f"Unknown element: {atomic_num}")
        data.node_type = torch.LongTensor(node_type_indices)

        # 构建全连接图的半边索引和标签
        # halfedge_index: 上三角索引 (i < j)
        halfedge_index = torch.triu_indices(num_atoms, num_atoms, offset=1)

        # 构建边类型矩阵（从真实边）
        edge_type_mat = torch.zeros([num_atoms, num_atoms], dtype=torch.long)
        if data.edge_index.numel() > 0:
            for k in range(data.edge_index.size(1)):
                i, j = data.edge_index[0, k].item(), data.edge_index[1, k].item()
                edge_type_mat[i, j] = data.edge_type[k].item()

        # 提取半边类型（标签）
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]

        data.halfedge_index = halfedge_index
        data.halfedge_type = halfedge_type

        # 删除不再需要的属性
        if hasattr(data, 'edge_index'):
            delattr(data, 'edge_index')
        if hasattr(data, 'edge_type'):
            delattr(data, 'edge_type')

        return data

    def decode_output(self, pred_node, pred_halfedge, halfedge_index):
        """
        从预测结果解码分子结构（2D版本，无坐标）
        """
        # 节点类型
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < self.num_element)

        if not isnot_masked_atom.all():
            edge_index_changer = -np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())

        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])

        # 边类型
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)
        edge_prob = np.max(pred_halfedge, axis=-1)

        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]

        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            bond_prob = bond_prob[~bond_for_masked_atom]

        # 转换为双向边
        bond_type = np.concatenate([bond_type, bond_type])
        bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)

        return {
            'element': element,
            'bond_type': bond_type,
            'bond_index': bond_index,
            'atom_prob': atom_prob,
            'bond_prob': bond_prob,
        }


class FeaturizeMol(object):
    def __init__(self, atomic_numbers, mol_bond_types,
                 use_mask_node, use_mask_edge):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)
        

        self.num_node_types = self.num_element + int(use_mask_node)
        self.num_edge_types = self.num_bond_types + 1 + int(use_mask_edge) # + 1 for the non-bonded edges
        self.use_mask_node = use_mask_node
        self.use_mask_edge = use_mask_edge
        
        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}
        
        
        self.follow_batch = ['node_type', 'halfedge_type']
        self.exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_list'
                             'bond_index', 'bond_type', 'num_bonds', 'num_atoms']
    
    def __call__(self, data: Drug3DData):
        # print("mol_bond_types",self.mol_bond_types)
        data.num_nodes = data.num_atoms
        
        # node type
        assert np.all([ele in self.atomic_numbers for ele in data.element]), 'unknown element'
        data.node_type = torch.LongTensor([self.ele_to_nodetype[ele.item()] for ele in data.element])
        
        # atom pos: sample a conformer from data.pos_all_confs; then move to origin
        idx = np.random.randint(data.pos_all_confs.shape[0])
        atom_pos = data.pos_all_confs[idx].float()
        atom_pos = atom_pos - atom_pos.mean(dim=0)

        data.node_pos = atom_pos
        data.i_conf = data.i_conf_list[idx]
        
        # build half edge (not full because perturb for edge_ij should be the same as edge_ji)
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.long)
        for i in range(data.num_bonds * 2):  # multiplication by two is for symmtric of bond index
            edge_type_mat[data.bond_index[0, i], data.bond_index[1, i]] = data.bond_type[i]
        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
        assert len(halfedge_type) == len(halfedge_index[0])
        
        data.halfedge_index = halfedge_index
        data.halfedge_type = halfedge_type
        assert (data.halfedge_type > 0).sum() == data.num_bonds
        
        return data
    
    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < self.num_element)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos[isnot_masked_atom]
        
        # get bond
        if self.num_edge_types == 1:
            return {
                'element': element,
                'atom_pos': atom_pos,
                'atom_prob': atom_prob,
            }
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        
        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)  # larger is mask type
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            bond_prob = bond_prob[~bond_for_masked_atom]

        bond_type = np.concatenate([bond_type, bond_type])
        bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            'element': element,
            'atom_pos': atom_pos,
            'bond_type': bond_type,
            'bond_index': bond_index,
            
            'atom_prob': atom_prob,
            'bond_prob': bond_prob,
        }
        
    
def make_data_placeholder(n_graphs, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:  # use statistics from GEOM-Drug dataset
        n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }


def collate_with_spectrum_features(data_list):
    """
    自定义数据整理函数，仅支持 formula / formula+dreams 两种模式

    输出 batch 字段（除 PyG 标准字段外）：
    - has_spectrum_mask: [B] bool，标记每个样本是否带 DreaMS 嵌入
    - batch_has_spectrum: bool，整批是否至少有一个样本带嵌入
    - pretrained_embedding_batch: [B, 1024] 或 None
    - instrument_type_idx_batch: [B]
    - ionization_type_idx_batch: [B]
    """
    from torch_geometric.data import Batch

    has_spectrum_data = any(getattr(data, 'has_spectrum', False) for data in data_list)
    has_pretrained_embeddings = any(
        getattr(data, 'pretrained_embedding', None) is not None for data in data_list
    )

    spectrum_info = []
    cleaned_data_list = []

    for data in data_list:
        info = {
            'has_spectrum': getattr(data, 'has_spectrum', False),
            'pretrained_embedding': getattr(data, 'pretrained_embedding', None),
            'instrument_type_idx': getattr(data, 'instrument_type_idx', 2),  # 默认 NONE
            'ionization_type_idx': getattr(data, 'ionization_type_idx', 0),  # 默认 [M+H]+
        }
        spectrum_info.append(info)

        data_copy = data.clone()
        attrs_to_remove = [
            'pretrained_embedding', 'has_spectrum',
            'instrument_type', 'instrument_type_idx', 'ionization_type_idx',
            'spectrum_index', 'original_mol_id', 'ms_file_path',
        ]
        for attr in attrs_to_remove:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)
        cleaned_data_list.append(data_copy)

    follow_batch = ['node_type', 'halfedge_type']
    exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_list',
                    'bond_index', 'bond_type', 'num_bonds', 'num_atoms']

    batch = Batch.from_data_list(cleaned_data_list, follow_batch=follow_batch, exclude_keys=exclude_keys)

    # 把每个样本的 smiles 单独挂到 batch（PyG Batch 的 exclude_keys 把 smiles 跳过了，
    # 但 align 阶段 multi_positive 需要按 smiles 分组 → 必须能拿到 list[str]）
    batch.smiles = [getattr(d, 'smiles', None) for d in data_list]

    has_spectrum_mask = []
    pretrained_embedding_list = []
    instrument_type_idx_list = []
    ionization_type_idx_list = []

    for info in spectrum_info:
        has_spectrum_mask.append(bool(info['has_spectrum']))
        if info['pretrained_embedding'] is not None:
            pretrained_embedding_list.append(info['pretrained_embedding'])
        else:
            pretrained_embedding_list.append(torch.zeros(1024))
        instrument_type_idx_list.append(info['instrument_type_idx'])
        ionization_type_idx_list.append(info['ionization_type_idx'])

    batch.batch_has_spectrum = bool(has_spectrum_data)
    batch.has_spectrum_mask = torch.tensor(has_spectrum_mask, dtype=torch.bool)

    if has_pretrained_embeddings:
        batch.pretrained_embedding_batch = torch.stack(pretrained_embedding_list)
    else:
        batch.pretrained_embedding_batch = None

    batch.instrument_type_idx_batch = torch.tensor(instrument_type_idx_list, dtype=torch.long)
    batch.ionization_type_idx_batch = torch.tensor(ionization_type_idx_list, dtype=torch.long)

    return batch


# ============================================================================
# DiffMSMSGDataset 专用 collate：输出 DeniMS 格式 + sparse 形式（后者给 BFN 主干用）
# ============================================================================

def collate_msg_diffms(data_list):
    """专给 DiffMSMSGDataset 用的 collate。

    输入：list of PyG Data，每个含
        x [N, 11], edge_index [2, M], edge_attr [M, 5],
        spec_sos [1, 13], spec_formula_array [128, 144], spec_mask_per_sample [129],
        smiles, mol_id, instrument_type_idx, ionization_type_idx

    输出 batch（一个 SimpleNamespace 风格对象）：
      DeniMS encoder 输入：
        spec_sos              [B, 1, 13]
        spec_formula_array    [B, 128, 144]
        spec_mask             [B, 129]   bool, True=padding
        dense_X               [B, N_max, 11]
        dense_E               [B, N_max, N_max, 5]   稀疏全连接图（缺边位置 = [1,0,0,0,0]，即 NoBond one-hot）
        dense_y               [B, 1]
        dense_node_mask       [B, N_max] bool

      BFN 主干 sparse 输入（与现有 train_flash 兼容）：
        node_type             [N_total]   long
        halfedge_index        [2, M_total_half]   只取上三角（无重复方向）
        halfedge_type         [M_total_half]      4 种键类型 + 0=无键
        node_type_batch       [N_total]
        halfedge_type_batch   [M_total_half]

      其他：
        smiles                list[str]
        instrument_type_idx_batch [B]
        ionization_type_idx_batch [B]
        has_spectrum_mask     [B] bool 全 True
        batch_has_spectrum    True
    """
    from torch_geometric.data import Batch
    from types import SimpleNamespace

    B = len(data_list)
    N_max = max(d.x.size(0) for d in data_list)

    # ---- DeniMS dense X / E / y / node_mask ----
    dense_X = torch.zeros(B, N_max, 11)
    dense_E = torch.zeros(B, N_max, N_max, 5)
    dense_y = torch.ones(B, 1)
    dense_node_mask = torch.zeros(B, N_max, dtype=torch.bool)

    for i, d in enumerate(data_list):
        n = d.x.size(0)
        dense_X[i, :n] = d.x
        dense_node_mask[i, :n] = True
        if d.edge_index.size(1) > 0:
            src, dst = d.edge_index
            dense_E[i, src, dst] = d.edge_attr
        # 把缺边位置（包括对角线）填 NoBond one-hot [1,0,0,0,0]
        edge_block = dense_E[i, :n, :n]
        zero_mask = (edge_block == 0).all(dim=-1)
        edge_block[zero_mask] = torch.tensor([1, 0, 0, 0, 0], dtype=dense_E.dtype)

    # ---- DeniMS spectrum 输入 ----
    spec_sos = torch.stack([d.spec_sos for d in data_list], dim=0)             # [B, 1, 13]
    spec_formula_array = torch.stack([d.spec_formula_array for d in data_list], dim=0)  # [B, 128, 144]
    spec_mask = torch.stack([d.spec_mask_per_sample for d in data_list], dim=0)  # [B, 129]

    # ---- BFN sparse 形式 (node_type, halfedge_index, halfedge_type) ----
    # DiffMS atom_types 顺序：B(0), C(1), N(2), O(3), F(4), Si(5), P(6), S(7), Cl(8), Br(9), I(10)
    # NEO BFN atomic_numbers=[5,6,7,8,9,14,15,16,17,33,34,35,53] → 索引 0..12
    # 重叠部分（B/C/N/O/F/Si/P/S/Cl）索引一致；Br: 9→11，I: 10→12（NEO 中间还有 As/Se）
    _DIFFMS2BFN = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12], dtype=torch.long)

    node_type_chunks = []
    halfedge_index_chunks = []
    halfedge_type_chunks = []
    node_type_batch_chunks = []
    halfedge_type_batch_chunks = []
    node_offset = 0
    for i, d in enumerate(data_list):
        n = d.x.size(0)
        node_type_chunks.append(_DIFFMS2BFN[d.x.argmax(dim=-1).long()])  # [n] in [0..12]
        node_type_batch_chunks.append(torch.full((n,), i, dtype=torch.long))

        # 把 PyG Data 的 (edge_index, edge_attr) 转半边
        # DeniMS 数据双向边都有，需要去重为半边
        if d.edge_index.size(1) > 0:
            src, dst = d.edge_index
            half_mask = src < dst
            half_src = src[half_mask] + node_offset
            half_dst = dst[half_mask] + node_offset
            halfedge_idx = torch.stack([half_src, half_dst], dim=0)
            half_type = d.edge_attr[half_mask].argmax(dim=-1).long()  # [m_half]
        else:
            halfedge_idx = torch.zeros(2, 0, dtype=torch.long)
            half_type = torch.zeros(0, dtype=torch.long)
        halfedge_index_chunks.append(halfedge_idx)
        halfedge_type_chunks.append(half_type)
        halfedge_type_batch_chunks.append(torch.full((half_type.size(0),), i, dtype=torch.long))
        node_offset += n

    node_type = torch.cat(node_type_chunks, dim=0)
    halfedge_index = torch.cat(halfedge_index_chunks, dim=1) if halfedge_index_chunks else torch.zeros(2, 0, dtype=torch.long)
    halfedge_type = torch.cat(halfedge_type_chunks, dim=0)
    node_type_batch = torch.cat(node_type_batch_chunks, dim=0)
    halfedge_type_batch = torch.cat(halfedge_type_batch_chunks, dim=0)

    # ---- 元信息 ----
    smiles = [getattr(d, 'smiles', None) for d in data_list]
    mol_ids = [getattr(d, 'mol_id', None) for d in data_list]
    instrument_idx = torch.tensor([int(d.instrument_type_idx) for d in data_list], dtype=torch.long)
    ionization_idx = torch.tensor([int(d.ionization_type_idx) for d in data_list], dtype=torch.long)

    # 用 SimpleNamespace 打包，方便 train_flash 像访问 PyG Batch 一样用 batch.xxx
    batch = SimpleNamespace(
        # DeniMS dense 输入
        dense_X=dense_X,
        dense_E=dense_E,
        dense_y=dense_y,
        dense_node_mask=dense_node_mask,
        # DeniMS spectrum 输入
        spec_sos=spec_sos,
        spec_formula_array=spec_formula_array,
        spec_mask=spec_mask,
        # BFN sparse
        node_type=node_type,
        halfedge_index=halfedge_index,
        halfedge_type=halfedge_type,
        node_type_batch=node_type_batch,
        halfedge_type_batch=halfedge_type_batch,
        # 元信息
        smiles=smiles,
        mol_ids=mol_ids,
        instrument_type_idx_batch=instrument_idx,
        ionization_type_idx_batch=ionization_idx,
        # 兼容字段（旧 train_flash 检查）
        has_spectrum_mask=torch.ones(B, dtype=torch.bool),
        batch_has_spectrum=True,
        num_graphs=B,
    )

    # 可以 .to(device) 通用化
    def _to(self, device):
        for attr_name in ['dense_X', 'dense_E', 'dense_y', 'dense_node_mask',
                          'spec_sos', 'spec_formula_array', 'spec_mask',
                          'node_type', 'halfedge_index', 'halfedge_type',
                          'node_type_batch', 'halfedge_type_batch',
                          'instrument_type_idx_batch', 'ionization_type_idx_batch',
                          'has_spectrum_mask']:
            v = getattr(self, attr_name, None)
            if torch.is_tensor(v):
                setattr(self, attr_name, v.to(device))
        return self
    batch.to = _to.__get__(batch, type(batch))

    return batch



# ============================================================================
# graph2mol 阶段 collate：SmilesDataset (sparse 13-class) → DeniMS dense (11-class) + 原 sparse
# ============================================================================

# NEO BFN atomic_numbers=[5,6,7,8,9,14,15,16,17,33,34,35,53] (13 类) → DeniMS atom_types 11 类
# 重叠：B/C/N/O/F/Si/P/S/Cl 直接对应；As(33)/Se(34) 在 DeniMS 中没有，丢分子；Br/I 索引偏移
# 返回 -1 表示该 atom DeniMS 不支持
_BFN2DIFFMS = torch.tensor([
    0,   # 0: B → 0
    1,   # 1: C → 1
    2,   # 2: N → 2
    3,   # 3: O → 3
    4,   # 4: F → 4
    5,   # 5: Si → 5
    6,   # 6: P → 6
    7,   # 7: S → 7
    8,   # 8: Cl → 8
    -1,  # 9: As → 不支持
    -1,  # 10: Se → 不支持
    9,   # 11: Br → 9
    10,  # 12: I → 10
], dtype=torch.long)


def collate_smiles_for_graph2mol(data_list):
    """graph2mol 阶段 collate（精简版，仅输出 BFN 主干所需的 sparse 张量）。

    历史包袱：之前还输出 dense_X/dense_E/dense_y/dense_node_mask 给 DeniMS graph_encoder 实时跑。
    但 graph2mol 阶段 graph_encoder 早就改成离线 cache（cond_emb_cached），不再实时 forward。
    所以 dense 部分对 BFN 训练没用，全部删掉，省下双层 Python 循环的开销（每 batch ~20 万次）。
    """
    from types import SimpleNamespace

    # 过滤掉含 As/Se 的分子（zmol cache 也不会有这些 SMILES）
    valid = []
    for d in data_list:
        if (_BFN2DIFFMS[d.node_type.long()] == -1).any().item():
            continue
        valid.append(d)
    if not valid:
        return None
    data_list = valid

    B = len(data_list)

    # BFN sparse
    node_type_chunks, halfedge_index_chunks, halfedge_type_chunks = [], [], []
    node_type_batch_chunks, halfedge_type_batch_chunks = [], []
    node_offset = 0
    for i, d in enumerate(data_list):
        n = int(d.node_type.size(0))
        node_type_chunks.append(d.node_type.long())
        node_type_batch_chunks.append(torch.full((n,), i, dtype=torch.long))
        m = int(d.halfedge_index.size(1))
        if m > 0:
            halfedge_index_chunks.append(d.halfedge_index + node_offset)
            halfedge_type_chunks.append(d.halfedge_type.long())
            halfedge_type_batch_chunks.append(torch.full((m,), i, dtype=torch.long))
        node_offset += n

    node_type = torch.cat(node_type_chunks, dim=0)
    halfedge_index = torch.cat(halfedge_index_chunks, dim=1) if halfedge_index_chunks else torch.zeros(2, 0, dtype=torch.long)
    halfedge_type = torch.cat(halfedge_type_chunks, dim=0) if halfedge_type_chunks else torch.zeros(0, dtype=torch.long)
    node_type_batch = torch.cat(node_type_batch_chunks, dim=0)
    halfedge_type_batch = torch.cat(halfedge_type_batch_chunks, dim=0) if halfedge_type_batch_chunks else torch.zeros(0, dtype=torch.long)

    smiles = [getattr(d, 'smiles', None) for d in data_list]

    batch = SimpleNamespace(
        node_type=node_type, halfedge_index=halfedge_index, halfedge_type=halfedge_type,
        node_type_batch=node_type_batch, halfedge_type_batch=halfedge_type_batch,
        smiles=smiles,
        instrument_type_idx_batch=torch.zeros(B, dtype=torch.long),
        ionization_type_idx_batch=torch.zeros(B, dtype=torch.long),
        num_graphs=B,
        spec_sos=None, spec_formula_array=None, spec_mask=None,
        has_spectrum_mask=torch.zeros(B, dtype=torch.bool),
        batch_has_spectrum=False,
    )

    def _to(self, device):
        for attr_name in ['node_type', 'halfedge_index', 'halfedge_type',
                          'node_type_batch', 'halfedge_type_batch',
                          'instrument_type_idx_batch', 'ionization_type_idx_batch',
                          'has_spectrum_mask']:
            v = getattr(self, attr_name, None)
            if torch.is_tensor(v):
                setattr(self, attr_name, v.to(device))
        return self
    batch.to = _to.__get__(batch, type(batch))

    return batch


# ============================================================================
# 缓存版 collate factory：closure 把 zmol/zms cache 包进去，给 batch 注入 cond_emb
# ============================================================================

def make_msg_diffms_collate_with_cache(zms_cache, zmol_cache=None):
    """ms2mol 阶段用：基础 collate_msg_diffms + 注入 cond_emb（按 spec_id 查 zms_cache）

    若样本 spec_id 不在 zms cache 里（极少数 spec 在 cache 构建时被跳过——例如 sub-formula JSON 为 null），
    直接过滤该样本，不报错。
    """
    def _collate(data_list):
        # 先过滤掉 cache 里没有的 spec
        filtered = [d for d in data_list
                    if getattr(d, 'mol_id', None) in zms_cache]
        if not filtered:
            return None
        batch = collate_msg_diffms(filtered)
        cond_emb_list = [zms_cache[mol_id].float() for mol_id in batch.mol_ids]
        batch.cond_emb_cached = torch.stack(cond_emb_list, dim=0)   # [B, 512]
        # 改 to 方法把 cond_emb_cached 也搬到 device
        orig_to = batch.to
        def _to_with_cond(self, device):
            orig_to(device)
            self.cond_emb_cached = self.cond_emb_cached.to(device)
            return self
        batch.to = _to_with_cond.__get__(batch, type(batch))
        return batch
    return _collate


def make_smiles_collate_with_cache(zmol_cache):
    """graph2mol 阶段用：基础 collate_smiles_for_graph2mol + 注入 cond_emb（按 SMILES 查 zmol_cache）

    若样本 SMILES 不在 zmol cache 里（极少数边角 SMILES 在 cache 构建时被 RDKit 跳过了），
    直接过滤该样本，不报错。
    """
    def _collate(data_list):
        # 先过滤掉 cache 里没有的 SMILES（包括没 smiles 字段的）
        filtered = [d for d in data_list
                    if getattr(d, 'smiles', None) in zmol_cache]
        if not filtered:
            return None
        batch = collate_smiles_for_graph2mol(filtered)
        if batch is None:
            return None
        cond_emb_list = [zmol_cache[smi].float() for smi in batch.smiles]
        batch.cond_emb_cached = torch.stack(cond_emb_list, dim=0)   # [B, 512]
        orig_to = batch.to
        def _to_with_cond(self, device):
            orig_to(device)
            self.cond_emb_cached = self.cond_emb_cached.to(device)
            return self
        batch.to = _to_with_cond.__get__(batch, type(batch))
        return batch
    return _collate
