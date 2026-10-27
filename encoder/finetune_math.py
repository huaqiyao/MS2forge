"""MSG/NPLIB1 fine-tuning loss."""
import torch
from torch.nn import functional as F

def compute_contrastive_loss(self, ms_emb, graph_emb, positive_mask=None):

    ms = F.normalize(ms_emb, dim=-1)
    gr = F.normalize(graph_emb, dim=-1)
    with torch.no_grad():
        self.inv_temperature.data.clamp_(min=1.0 / 200.0)
    inv_temp = self.inv_temperature
    logits_ms = ms @ gr.t() / inv_temp.clamp(min=0.001)
    logits_gr = logits_ms.t()
    B = ms.size(0)
    device = ms.device
    if positive_mask is None:
        labels = torch.arange(B, device=device)
        loss_ms = F.cross_entropy(logits_ms, labels)
        loss_gr = F.cross_entropy(logits_gr, labels)
        return 0.5 * (loss_ms + loss_gr)
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
