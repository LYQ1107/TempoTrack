"""Explicit losses used by the research methods."""

from .identity import identity_contrastive_loss
from .predictive import predictive_memory_loss
from .regularization import vicreg_regularization
from .flow import flow_matching_loss
from .diffusion import epsilon_loss
from .policy import masked_behavior_cloning_loss, ppo_loss

__all__ = ["epsilon_loss", "flow_matching_loss", "identity_contrastive_loss", "masked_behavior_cloning_loss", "ppo_loss", "predictive_memory_loss", "vicreg_regularization"]
