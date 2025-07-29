from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn
from care.evaluators.ocp import OCPIntermediateEvaluator, OCPReactionEvaluator
from care.evaluators.mace import MACEIntermediateEvaluator, MACEReactionEvaluator
from care.evaluators.petmad import PETMADIntermediateEvaluator, PETMADReactionEvaluator
from care.evaluators.orb import ORBIntermediateEvaluator, ORBReactionEvaluator
from care.evaluators.sevennet import SevenNetIntermediateEvaluator, SevenNetReactionEvaluator

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn),
    "ocp": (OCPIntermediateEvaluator, OCPReactionEvaluator),
    "mace": (MACEIntermediateEvaluator, MACEReactionEvaluator),
    "petmad": (PETMADIntermediateEvaluator, PETMADReactionEvaluator),
    "orb": (ORBIntermediateEvaluator, ORBReactionEvaluator),
    "sevennet": (SevenNetIntermediateEvaluator, SevenNetReactionEvaluator),
}

def get_available_evaluators():
    """
    Show available energy evaluators in CARE.
    """
    return list(eval_dict.keys())

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
]
