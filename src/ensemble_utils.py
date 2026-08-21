# ensemble_utils.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

@torch.no_grad()
def predict_proba_single(model: torch.nn.Module,
                         img_tensor: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """
    img_tensor: shape [1, C, H, W], already transformed.
    Returns: probs tensor [num_classes]
    """
    model.eval().to(device)
    logits = model(img_tensor.to(device))
    probs = F.softmax(logits.squeeze(0), dim=-1)
    return probs.detach().cpu()

@torch.no_grad()
def predict_proba_ensemble(models: List[torch.nn.Module],
                           img_tensor: torch.Tensor,
                           device: torch.device,
                           weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """
    Soft/weighted voting over models.
    Returns dict with:
      - 'individual_probs': list of [num_classes] tensors
      - 'ensemble_probs': [num_classes] tensor
    """
    if weights is None:
        weights = [1.0] * len(models)
    assert len(weights) == len(models), "weights length must match number of models"

    individual_probs = []
    wsum = sum(weights)
    accum = None

    for m, w in zip(models, weights):
        p = predict_proba_single(m, img_tensor, device)  # [num_classes]
        individual_probs.append(p)
        accum = p * w if accum is None else accum + p * w

    ensemble_probs = accum / wsum
    return {
        "individual_probs": individual_probs,
        "ensemble_probs": ensemble_probs
    }

def topk_from_probs(probs: torch.Tensor, k: int = 3) -> List[Tuple[int, float]]:
    """
    probs: [num_classes]
    Returns list of (class_idx, prob) sorted desc by prob.
    """
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for v, i in zip(vals, idxs)]
