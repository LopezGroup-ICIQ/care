from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn
from care.evaluators.ocp import OCPIntermediateEvaluator
from care.evaluators.mace import MACEIntermediateEvaluator
from care.evaluators.petmad import PETMADIntermediateEvaluator
from care.evaluators.orb import ORBIntermediateEvaluator
from care.evaluators.sevennet import SevenNetIntermediateEvaluator
from care.evaluators.reaction_estimators import BarrierlessReactionEnergyEstimator, NEBReactionEnergyEstimator

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn),
    "ocp": (OCPIntermediateEvaluator, BarrierlessReactionEnergyEstimator),
    "mace": (MACEIntermediateEvaluator, BarrierlessReactionEnergyEstimator),
    "petmad": (PETMADIntermediateEvaluator, BarrierlessReactionEnergyEstimator),
    "orb": (ORBIntermediateEvaluator, BarrierlessReactionEnergyEstimator),
    "sevennet": (SevenNetIntermediateEvaluator, BarrierlessReactionEnergyEstimator),
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
