"""
Base class for energy estimators.
"""

from typing import Optional
from abc import ABC, abstractmethod

from care import Intermediate, Surface, ElementaryReaction

class IntermediateEnergyEstimator(ABC):
    """
    Base class for intermediate energy estimators.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def eval(self,
             inter: Intermediate, 
             surf: Optional[Surface] = None) -> None:
        """
        Estimate the energy of a state.

        Args:
            inter (Intermediate): The intermediate.
            surf (Surface, optional): The surface. Defaults to None.
        """
        pass

class ReactionEnergyEstimator(ABC):
    """
    Base class for intermediate energy estimators.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, reaction: ElementaryReaction) -> None:
        """
        Estimate the energy of a state.

        Args:
            inter (Intermediate): The intermediate.
            surf (Surface, optional): The surface. Defaults to None.
        """
        pass