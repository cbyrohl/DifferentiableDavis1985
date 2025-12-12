"""
DifferentiableDavis1985

Reproducing the Davis et al. 1985 CDM simulations using differentiable forward models.
"""

from differentiabledavis1985.forward_model import (
    BaseSimulation,
    Davis1985Simulation,
)
from differentiabledavis1985.davis1985_models import (
    Fig1BottomLeftSimulation,
    EdS1Simulation,
)
from differentiabledavis1985.losses import (
    chi2,
    chi2_log,
    reduced_chi2,
    reduced_chi2_log,
)

__all__ = [
    "BaseSimulation",
    "Davis1985Simulation",
    "Fig1BottomLeftSimulation",
    "EdS1Simulation",
    "chi2",
    "chi2_log",
    "reduced_chi2",
    "reduced_chi2_log",
]
