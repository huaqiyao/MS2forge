"""One convention for node relabelling and induced unordered edge bijections."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Relabel:
    new_to_old_node: torch.Tensor
    new_to_old_edge: torch.Tensor
    old_to_new_node: torch.Tensor
    old_to_new_edge: torch.Tensor
    edges: torch.Tensor

def relabel(perm, old_edges=None):
    perm = torch.as_tensor(perm, dtype=torch.long, device='cpu')
    n = len(perm)
    if not torch.equal(torch.sort(perm).values, torch.arange(n)):
        raise ValueError('Node permutation is not a bijection')
    old_edges = torch.triu_indices(n, n, 1) if old_edges is None else old_edges.cpu().long()
    validate_edges(old_edges, n)
    new_edges = torch.triu_indices(n, n, 1)
    lookup = torch.full((n, n), -1, dtype=torch.long)
    ids = torch.arange(old_edges.shape[1])
    lookup[old_edges[0], old_edges[1]] = ids
    lookup[old_edges[1], old_edges[0]] = ids
    mapped = perm[new_edges]
    edge_perm = lookup[mapped[0], mapped[1]]
    inverse_node = torch.empty_like(perm); inverse_node[perm] = torch.arange(n)
    inverse_edge = torch.empty_like(edge_perm); inverse_edge[edge_perm] = torch.arange(len(edge_perm))
    return Relabel(perm, edge_perm, inverse_node, inverse_edge, new_edges)

def validate_edges(edges, n):
    if edges.shape != (2, n*(n-1)//2):
        raise ValueError('Complete unordered edge count mismatch')
    if edges.numel() and ((edges<0).any() or (edges>=n).any() or (edges[0]==edges[1]).any()):
        raise ValueError('Invalid edge endpoints')
    keys = torch.minimum(edges[0],edges[1])*n + torch.maximum(edges[0],edges[1])
    if keys.unique().numel()!=len(keys):
        raise ValueError('Duplicate unordered edge state')

def permute_supervision(node, edges, labels, perm, node_features=None, edge_features=None):
    r = relabel(perm, edges)
    return {'node_type':node[r.new_to_old_node], 'halfedge_index':r.edges,
            'halfedge_type':labels[r.new_to_old_edge], 'mapping':r,
            'node_features':None if node_features is None else node_features[r.new_to_old_node],
            'edge_features':None if edge_features is None else edge_features[r.new_to_old_edge]}
