from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn
from care.evaluators.fairchemv1 import FairChemV1IntermediateEvaluator
from care.evaluators.mace import MACEIntermediateEvaluator
from care.evaluators.upet import UPETIntermediateEvaluator
from care.evaluators.orb import ORBIntermediateEvaluator
from care.evaluators.sevennet import SevenNetIntermediateEvaluator
from care.evaluators.reaction_estimators import BarrierlessReactionEnergyEstimator, NEBReactionEnergyEstimator

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn),
    "fairchemv1": (FairChemV1IntermediateEvaluator, NEBReactionEnergyEstimator),
    "mace": (MACEIntermediateEvaluator, NEBReactionEnergyEstimator),
    "upet": (UPETIntermediateEvaluator, NEBReactionEnergyEstimator),
    "orb": (ORBIntermediateEvaluator, NEBReactionEnergyEstimator),
    "sevennet": (SevenNetIntermediateEvaluator, NEBReactionEnergyEstimator),
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

def load_reaction_evaluator(model: str, mlp: IntermediateEnergyEstimator = None, ts_eval: bool = True, **kwargs) -> ReactionEnergyEstimator:
    """
    Load the reaction evaluator.

    Args:
        name (str): The name of the reaction evaluator.

    Returns:
        ReactionEnergyEstimator: The reaction evaluator.
    """
    if model == "gamenetuq":
        return eval_dict[model][1](**kwargs) if ts_eval else BarrierlessReactionEnergyEstimator(**kwargs)
    return eval_dict[model][1](mlp=mlp, **kwargs) if ts_eval else BarrierlessReactionEnergyEstimator(**kwargs)

__all__ = [
    "IntermediateEnergyEstimator",
    "ReactionEnergyEstimator",
    "load_inter_evaluator",
    "load_reaction_evaluator",
]
