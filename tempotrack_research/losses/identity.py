"""Identity losses that tolerate multiple positives and unknown labels."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def identity_contrastive_loss(embeddings: Tensor, identities: Tensor, known_mask: Tensor | None = None, temperature: float = 0.07) -> Tensor:
    if embeddings.ndim != 2 or identities.ndim != 1:
        raise ValueError("embeddings must be [N,D] and identities [N]")
    mask = identities >= 0 if known_mask is None else known_mask.bool() & (identities >= 0)
    if mask.sum() < 2:
        raise ValueError("identity loss has fewer than two known examples")
    z = F.normalize(embeddings[mask], dim=-1)
    y = identities[mask]
    logits = z @ z.t() / max(float(temperature), 1e-4)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    positives = (y[:, None] == y[None, :]) & ~eye
    valid = positives.any(dim=-1)
    if not valid.any():
        raise ValueError("identity loss has no positive pair")
    log_prob = logits.log_softmax(dim=-1)
    return -(log_prob * positives.to(log_prob.dtype)).sum(dim=-1)[valid].div(positives.sum(dim=-1)[valid]).mean()
