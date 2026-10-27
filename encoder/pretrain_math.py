"""Pretraining losses and padded molecular graph batching."""
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from tqdm import tqdm

def batch_graphs_to_padded_data(data_list, device="cuda"):
    """
    Convert a list of PyTorch Geometric Data objects to a single padded batch.
    
    Takes:
        data_list: List of Data objects, each containing node features (x), 
                   edge indices (edge_index), and edge attributes (edge_attr)

    Returns:
        Data object with padded tensors:
        - X: [B, max_nodes, node_feat_dim] node features
        - E: [B, max_nodes, max_nodes, edge_feat_dim] edge features
        - y: [B, 1] dummy target
        - node_mask: [B, max_nodes] boolean mask for valid nodes
    """
    B = len(data_list)
    max_nodes = max(data.x.size(0) for data in data_list)
    node_feat_dim = data_list[0].x.size(1)
    edge_feat_dim = data_list[0].edge_attr.size(1)

    # Allocate padded tensors
    X = torch.zeros(B, max_nodes, node_feat_dim)
    E = torch.zeros(B, max_nodes, max_nodes, edge_feat_dim)
    y = torch.ones(X.shape[0], 1).type_as(X)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    atom_counts = torch.zeros(B, dtype=torch.long)

    for i, data in enumerate(data_list):
        n = data.x.size(0)
        X[i, :n] = data.x
        node_mask[i, :n] = 1
        atom_counts[i] = n

        src, dst = data.edge_index
        E[i, src, dst] = data.edge_attr

        # Set zero edges to default edge type
        valid_n = n  
        edge_block = E[i, :valid_n, :valid_n]  
        zero_mask = (edge_block == 0).all(dim=-1)  
        edge_block[zero_mask] = torch.tensor([1, 0, 0, 0, 0], dtype=E.dtype, device=E.device)

    return Data(
        X=X.to(device),
        E=E.to(device),
        y=y.to(device),
        node_mask=node_mask.to(device)
    )

def compute_contrastive_logits(ms_features, mol_features, temperature=15, calc_similarity=False):

    # L2 normalize features
    ms_features = ms_features / ms_features.norm(dim=1, keepdim=True)
    mol_features = mol_features / mol_features.norm(dim=1, keepdim=True)

    # Compute similarity matrix
    logits_per_ms = temperature * ms_features @ mol_features.t()       
    logits_per_mol = logits_per_ms.t()

    if calc_similarity:
        cos_sim = (ms_features * mol_features).sum(dim=1)
        average_cosine_similarity = cos_sim.mean().item()
        return logits_per_ms, logits_per_mol, average_cosine_similarity
    
    return logits_per_ms, logits_per_mol

def evaluate_during_training(model, val_dataloader, graph_dict, args, eval_part: float = 1.0, epoch=None):

    model.eval()

    mode = getattr(args, "mode", None)

    if mode == "mixed":
        mode = "contrastive"
        mixed = True
    else:
        mixed = False

    if mode == 'contrastive':
        loss_fn = nn.CrossEntropyLoss()
        use_graph = True
        sigmoid = None
    elif mode == 'fp':
        loss_fn = nn.MSELoss()
        sigmoid = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_batches = len(val_dataloader)
    batches_num = max(1, int(total_batches * float(eval_part)))

    total_samples = 0
    total_acc = 0
    total_loss = 0.0
    total_cosine_similarity = 0.0

    desc = f"Val Epoch {epoch}" if epoch is not None else "Validation evaluation"
    pbar = tqdm(range(batches_num), desc=desc)
    data_iter = iter(val_dataloader)

    for _ in pbar:
        sos, formula_array, mask, smiles, _ = next(data_iter)

        if mode == 'contrastive':
            # Build graph batch
            graph_list = [graph_dict[smi][0] for smi in smiles]
            graph_batch = batch_graphs_to_padded_data(graph_list, device=args.device)
            
            if mixed:
                ms_features, graph_features, _ = model(sos, formula_array, graph=graph_batch, mask=mask)
            else:
                ms_features, graph_features = model(sos, formula_array, graph=graph_batch, mask=mask)          

        else:  # mode == 'fp'
            # Get fingerprint labels
            fp_features = torch.cat(
                [graph_dict[smi][1] for smi in smiles], dim=0
            ).to(args.device).float()

            preds = model(sos, formula_array, mask=mask)
            ms_features = sigmoid(preds)
            graph_features = fp_features

        # Compute logits and similarity
        temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
        logits_per_ms, logits_per_mol, avg_cosine = compute_contrastive_logits(
            ms_features, graph_features, temp, calc_similarity=True
        )

        # Top-1 predictions
        ms_class = torch.argmax(logits_per_ms, dim=1)
        mol_class = torch.argmax(logits_per_mol, dim=1)

        # Ground-truth labels (diagonal alignment)
        labels = torch.arange(ms_class.shape[0], device=args.device)

        # Compute loss
        if mode == 'contrastive':
            loss = (loss_fn(logits_per_ms, labels) + loss_fn(logits_per_mol, labels)) / 2.0
        else:  # fp
            loss = loss_fn(ms_features, graph_features)

        # Compute accuracy
        acc_1 = torch.sum(ms_class.eq(labels))
        acc_2 = torch.sum(mol_class.eq(labels))

        batch_size = labels.shape[0]
        total_loss += float(loss.item())
        total_acc += int((acc_1 + acc_2).item())
        total_samples += batch_size * 2  # two directions
        total_cosine_similarity += float(avg_cosine) * batch_size

    metrics = {
        'Val/eval_acc': total_acc / total_samples,
        'Val/eval_loss': total_loss / batches_num,
        'Val/cosine_similarity': 2 * total_cosine_similarity / total_samples,
    }
    return metrics

def warm_cosine_schedule(step, warmsteps, total_iterations):
    if step < warmsteps:
        return step / warmsteps  # Linear warm-up
    else:
        return 0.5 * (1 + np.cos(np.pi * (step - warmsteps) / (total_iterations - warmsteps)))

def build_scheduler(optim, total_iterations, scheduler_type='WarmCosine', warmsteps=200):
    if scheduler_type == 'WarmCosine':
        return lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: warm_cosine_schedule(s, warmsteps, total_iterations))
    elif scheduler_type == 'Cosine':
        return lr_scheduler.CosineAnnealingLR(optim, T_max=total_iterations, eta_min=1e-7)
    elif scheduler_type == 'Constant':
        return torch.optim.lr_scheduler.ConstantLR(optim)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
