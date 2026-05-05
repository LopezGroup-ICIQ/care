from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn, GAMENETUQ_AVAILABLE
from care.evaluators.fairchemv1 import FairChemV1IntermediateEvaluator, FAIRCHEMV1_AVAILABLE
from care.evaluators.fairchemv2 import FairChemV2IntermediateEvaluator, FAIRCHEMV2_AVAILABLE
from care.evaluators.mace import MACEIntermediateEvaluator, MACE_AVAILABLE
from care.evaluators.upet import UPETIntermediateEvaluator, UPET_AVAILABLE
from care.evaluators.orb import ORBIntermediateEvaluator, ORB_AVAILABLE
from care.evaluators.sevennet import SevenNetIntermediateEvaluator, SEVENNET_AVAILABLE
from care.evaluators.reaction_estimators import BarrierlessReactionEnergyEstimator, NEBReactionEnergyEstimator

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn, GAMENETUQ_AVAILABLE),
    "fairchemv1": (FairChemV1IntermediateEvaluator, NEBReactionEnergyEstimator, FAIRCHEMV1_AVAILABLE),
    "fairchemv2": (FairChemV2IntermediateEvaluator, NEBReactionEnergyEstimator, FAIRCHEMV2_AVAILABLE),
    "mace": (MACEIntermediateEvaluator, NEBReactionEnergyEstimator, MACE_AVAILABLE),
    "upet": (UPETIntermediateEvaluator, NEBReactionEnergyEstimator, UPET_AVAILABLE),
    "orb": (ORBIntermediateEvaluator, NEBReactionEnergyEstimator, ORB_AVAILABLE),
    "sevennet": (SevenNetIntermediateEvaluator, NEBReactionEnergyEstimator, SEVENNET_AVAILABLE),
}

def get_available_evaluators(installed_only=False) -> list:
    """
    Show available energy evaluators in CARE.
    Args:
        installed_only (bool): If True, only show installed evaluators. If False, show all evaluators.
    """
    if installed_only:
        return [key for key, (inter, rxn, available) in eval_dict.items() if available]
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
