"""Controllable Visual Dynamics Memory (CVDM).

This package contains the rover-side world-model stack used to ground visual
novelty in action-conditioned, controllable transitions.
"""

from CVDM.config import CVDMConfig
from CVDM.models import ControllableVisualDynamics
from CVDM.training import CVDMTrainer

__all__ = ["CVDMConfig", "ControllableVisualDynamics", "CVDMTrainer"]
