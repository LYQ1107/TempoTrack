"""Small candidate scorer with a shared, label-free feature definition."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

EVIDENCE_COLUMNS = (
    "det_score", "current_fast_cosine", "current_slow_cosine", "fast_slow_cosine",
    "fragment_age_clipped_over_100", "previous_gap_clipped_over_max_gap",
    "log_area_ratio_clipped_over_2",
)
FEATURE_NAMES = (
    "cosine_top1", "cosine_top3_mean", "cosine_top5_mean", "cosine_mean", "cosine_std",
    "query_support_mean", "query_support_std", "query_support_min", "query_support_max",
    "memory_length_over_64", "anchor_score_mean", "anchor_score_std", "anchor_fast_mean",
    "anchor_slow_mean", "anchor_fast_slow_mean", "log1p_gap", "gap_over_max_gap",
    "prefilter_rank", "inverse_prefilter_rank", "raw_support", "raw_minus_best_other",
    "group_best_second_margin", "raw_rank_percentile", "candidate_count",
)


def candidate_features(cosine, evidence, gap, rank, *, top_r=3, max_gap=360):
    """Only valid query/memory rows enter; callers remove all padding first."""
    c = np.asarray(cosine, dtype=np.float32)
    e = np.asarray(evidence, dtype=np.float32)
    if c.ndim != 2 or min(c.shape) < 1 or e.shape != (c.shape[1], 7):
        raise ValueError("invalid valid-query/memory evidence shapes")
    if not np.isfinite(c).all() or not np.isfinite(e).all() or gap <= 0 or rank < 1:
        raise ValueError("nonfinite features or noncausal candidate")
    flat = np.sort(c.reshape(-1))
    supports = np.sort(c, axis=1)[:, -min(top_r, c.shape[1]):].mean(axis=1)
    return np.asarray([
        flat[-1], flat[-min(3,len(flat)):].mean(), flat[-min(5,len(flat)):].mean(), flat.mean(), flat.std(),
        supports.mean(), supports.std(), supports.min(), supports.max(),
        c.shape[1]/64, e[:,0].mean(), e[:,0].std(), e[:,1].mean(), e[:,2].mean(), e[:,3].mean(),
        np.log1p(gap), gap/max_gap, rank, 1/rank,
    ], dtype=np.float32)


def add_event_context(features):
    """Event context uses appearance only, including at training time."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 19 or not len(x):
        raise ValueError("expected a nonempty event of 19 independent features")
    raw = x[:,5]
    order = np.argsort(-raw, kind="stable")
    second = float(raw[order[1]]) if len(x)>1 else float(raw[order[0]])
    best = float(raw[order[0]])
    other = np.full(len(x), best, dtype=np.float32); other[order[0]] = second
    percentile = np.empty(len(x), dtype=np.float32)
    percentile[order] = np.arange(len(x))/max(1,len(x)-1)
    context = np.column_stack((raw,raw-other,np.full(len(x),best-second),percentile,np.full(len(x),len(x))))
    return np.concatenate((x,context),axis=1).astype(np.float32)


class CandidateReranker(nn.Module):
    def __init__(self, input_dim=len(FEATURE_NAMES), hidden=(64,32), dropout=0.1):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.network = nn.Sequential(nn.Linear(input_dim,hidden[0]),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden[0],hidden[1]),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden[1],1))

    def forward(self, features):
        return self.network((features-self.feature_mean)/self.feature_scale).squeeze(-1)


def group_ranking_loss(logits, labels, mask=None, margin=0.2):
    """Multi-positive logsumexp ranking + 0.2 hardest-negative margin."""
    if logits.ndim == 1:
        logits, labels = logits[None], labels[None]
        mask = None if mask is None else mask[None]
    valid = (labels>=0) if mask is None else mask.bool() & (labels>=0)
    positive = valid & (labels==1); negative = valid & (labels==0)
    usable = positive.any(dim=1) & negative.any(dim=1)
    if not bool(usable.any()):
        raise ValueError("rank loss requires a positive and a labeled negative per event")
    z=logits[usable]; p=positive[usable]; n=negative[usable]; v=valid[usable]
    rank = torch.logsumexp(z.masked_fill(~v,-torch.inf),dim=1)-torch.logsumexp(z.masked_fill(~p,-torch.inf),dim=1)
    hard = torch.relu(margin+z.masked_fill(~n,-torch.inf).max(dim=1).values-z.masked_fill(~p,-torch.inf).max(dim=1).values)
    return (rank+0.2*hard).mean()
