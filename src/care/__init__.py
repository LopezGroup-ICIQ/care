import pathlib as pl

from care.crn.intermediate import Intermediate
from care.crn.surface import Surface
from care.crn.elementary_reaction import ElementaryReaction
from care.crn.reaction_network import ReactionNetwork

__all__ = ['Intermediate', 'ElementaryReaction', 'ReactionNetwork', 'Surface']
__version__ = '0.0.1'

MODULEROOT = pl.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/data/metal_surfaces.db"
MODEL_PATH = f"{MODULEROOT}/gnn/gamenet_uq_dim192"