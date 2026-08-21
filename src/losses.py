import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional per-class alpha weights.

    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # logits: [N,K], target: [N]
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        if self.label_smoothing > 0.0:
            K = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (K - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_prob).sum(dim=1)
            pt = (true_dist * prob).sum(dim=1)
        else:
            ce = F.nll_loss(log_prob, target, reduction="none")
            pt = prob.gather(1, target.unsqueeze(1)).squeeze(1)

        focal = (1 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[target]
            else:
                alpha_t = torch.tensor(self.alpha, device=logits.device)
            focal = alpha_t * focal

        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "mean":
            return focal.mean()
        return focal
