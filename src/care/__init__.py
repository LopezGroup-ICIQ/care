# import juliacall  # to avoid segfaults
from pickle import load, dump

from care.constants import *
from care.crn.surface import Surface
from care.crn.intermediate import Intermediate
from care.crn.elementary_reaction import ElementaryReaction, ReactionMechanism
from care.crn.reaction_network import ReactionNetwork
from care.crn.utils.blueprint import gen_blueprint
from care.crn.templates.dissociation import dissociate

def load_crn(file_path: str) -> ReactionNetwork:
    with open(file_path, "rb") as f:
        return load(f)
    
def save_crn(crn: ReactionNetwork, file_path: str):
    with open(file_path, "wb") as f:
        dump(crn, f)

__all__ = [
    "Intermediate",
    "ElementaryReaction",
    "ReactionNetwork",
    "Surface",
    "ReactionMechanism",
    "gen_blueprint",
    "dissociate",
    "load_crn", 
    "save_crn",
]
__version__ = "1.0.0"
