import pathlib as pl

from care.crn.surface import Surface
from care.crn.intermediate import Intermediate
from care.crn.elementary_reaction import ElementaryReaction
from care.crn.reaction_network import ReactionNetwork
from care.crn.energy_estimator import ReactionEnergyEstimator, IntermediateEnergyEstimator

__all__ = ["Intermediate", "ElementaryReaction", "ReactionNetwork", "Surface", "ReactionEnergyEstimator", "IntermediateEnergyEstimator"]
__version__ = "0.0.1"

MODULEROOT = pl.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/data/metal_surfaces.db"
MODEL_PATH = f"{MODULEROOT}/gnn/dim192_5splits"
