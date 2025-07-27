# import juliacall  # to avoid segfaults
from pickle import load, dump
import re

def format_reaction(s: str) -> str:
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    def handle_stoichiometry(match):
        coeff = int(match.group(1))
        molecule = match.group(2)
        return molecule if coeff == 1 else f"{coeff}{molecule}"

    s = re.sub(r"\[(\d+)\]([A-Za-z0-9*]+)", handle_stoichiometry, s)
    def subscript_replacer(match):
        letters = match.group(1)
        digits = match.group(2)
        return letters + digits.translate(subscript_map)

    return re.sub(r"([A-Za-z])(\d+)", subscript_replacer, s)

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
__version__ = "0.1.0"
