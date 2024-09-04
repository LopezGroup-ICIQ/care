from ase.db import connect

from care import Surface

def load_surface(metal: str, hkl: str) -> Surface:
    """
    Load surface from ASE database.

    Args:
        metal (str): Metal symbol (e.g., "Ag")
        hkl (str): Miller index (e.g., "111", "0001")

    Note:
        The database should contain a surface with the given metal and Miller index.
        For hcp metals, the Miller index should be in the form "hkil", negative indices
        should be written as "mh-kil" (e.g. "10m11" stands for 10-11).
    """
    metal_db = connect(DB_PATH)
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    try:
        surface_ase = metal_db.get_atoms(
            calc_type="surface", metal=metal, facet=metal_structure, add_additional_information=True
        )
    except:
        # Generate surface from scratch (possible with current implementation!!!)
        raise ValueError(f"{metal} surface {metal_structure} not found in the database.")

    return Surface(surface_ase, hkl)

from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn
from care.evaluators.oc20models import OC20IntermediateEvaluator, OC20ReactionEvaluator
from care.evaluators.mace_mp_0 import MaceIntermediateEvaluator, MaceReactionEvaluator
from care.evaluators.gamenet_uq import DB_PATH, METAL_STRUCT_DICT

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn),
    "oc20models": (OC20IntermediateEvaluator, OC20ReactionEvaluator),
    "macemp0": (MaceIntermediateEvaluator, MaceReactionEvaluator),
}

def load_inter_evaluator(model: str, surface, **kwargs) -> IntermediateEnergyEstimator:
    """
    Load the intermediate evaluator.

    Args:
        name (str): The name of the intermediate evaluator.

    Returns:
        IntermediateEnergyEstimator: The intermediate evaluator.
    """
    return eval_dict[model][0](surface, **kwargs)

def load_reaction_evaluator(model: str, intermediates, **kwargs) -> ReactionEnergyEstimator:
    """
    Load the reaction evaluator.

    Args:
        name (str): The name of the reaction evaluator.

    Returns:
        ReactionEnergyEstimator: The reaction evaluator.
    """
    return eval_dict[model][1](intermediates, **kwargs)

__all__ = [
    "IntermediateEnergyEstimator",
    "ReactionEnergyEstimator",
    "load_inter_evaluator",
    "load_reaction_evaluator",
    "load_surface",
]
