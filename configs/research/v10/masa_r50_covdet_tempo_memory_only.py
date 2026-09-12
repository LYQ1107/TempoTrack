"""MASA-R50 + COV public detections with the reranker-disabled overlay."""

_base_ = ['./masa_r50_covdet_native.py']

model = dict(
    tracker=dict(
        tempo=dict(
            enabled=True,
            alpha_fast=0.70,
            alpha_slow=0.15,
            min_gap=0,
            max_gap=60,
            candidate_top_k=8,
            top_r=3,
            native_weight=0.45,
            active_weight=0.35,
            support_weight=0.20,
            gap_penalty=0.02,
            score_threshold=0.60,
            margin_threshold=0.0,
            reliability_weight=0.0,
            memory_capacity=64,
            reranker_weight=0.0,
            reranker_checkpoint=None,
            reranker_source_root=None,
            reranker_device='cpu',
        )
    )
)
