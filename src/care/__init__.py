from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
import logging
import os
import pathlib
import re
import shutil
import sys
import warnings

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
JULIA_ENV_PATH = CURRENT_DIR / "julia_env"
os.environ["PYTHON_JULIACALL_PROJECT"] = str(JULIA_ENV_PATH)

def setup_julia():
    if shutil.which("julia") is None:
        raise RuntimeError(
            "Julia executable not found in PATH.\n"
            "Please install Julia 1.11: https://julialang.org/downloads/\n"
            "Or run: curl -fsSL https://install.julialang.org | sh"
        )

@contextmanager
def silent_context(suppress_stdout=True, suppress_logging=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if suppress_logging:
            previous_log_level = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
        old_stdout = sys.stdout
        if suppress_stdout:
            sys.stdout = open(os.devnull, 'w')
            
        try:
            yield
        finally:
            if suppress_stdout:
                sys.stdout.close()
                sys.stdout = old_stdout
            if suppress_logging:
                logging.disable(previous_log_level)

def format_reaction(s: str) -> str:
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    superscript_map = str.maketrans("+-", "⁺⁻")

    def handle_stoichiometry(match):
        coeff = int(match.group(1))
        molecule = match.group(2)
        return molecule if coeff == 1 else f"{coeff}{molecule}"
    s = re.sub(r"\[(\d+)\]([A-Za-z0-9*()+-]+)", handle_stoichiometry, s)
    s = re.sub(r"\(([^)]+)\)", lambda m: m.group(0).translate(subscript_map), s)

    def subscript_replacer(match):
        letters = match.group(1)
        digits = match.group(2)
        return letters + digits.translate(subscript_map)
    s = re.sub(r"([A-Za-z])(\d+)", subscript_replacer, s)
    s = re.sub(r"([A-Za-z0-9₀₁₂₃₄₅₆₇₈₉₍₎]+)([+-])", lambda m: m.group(1) + m.group(2).translate(superscript_map), s)
    return s

from care.constants import *
from care.crn.surface import Surface, load_surface
from care.crn.intermediate import Intermediate
from care.crn.elementary_reaction import ElementaryReaction, ReactionMechanism
from care.crn.reaction_network import ReactionNetwork
from care.crn.utils.blueprint import gen_blueprint
from care.crn.templates.dissociation import dissociate        

__all__ = [
    "Intermediate",
    "ElementaryReaction",
    "ReactionNetwork",
    "Surface",
    "ReactionMechanism",
    "gen_blueprint",
    "dissociate",
    "load_surface",
]

try:
    __version__ = version("care-crn")
except PackageNotFoundError:
    __version__ = "unknown"
