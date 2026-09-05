"""Optional MASA tracker override for the M1 causal memory path.

Include this mapping in an existing MASA detector config only after a trained
predictive-controller checkpoint has been produced.  The default legacy
configs remain fixed dual EMA.
"""

tracker_override = dict(
    memory_mode="predictive_dual",
    memory_controller_hidden=128,
    memory_confidence_threshold=0.55,
)
