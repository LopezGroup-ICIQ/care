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
    def __call__(self, inter: Intermediate, surf: Optional[Surface] = None) -> None:
        self.eval(inter, surf)

    @property
    @abstractmethod
    def adsorbate_domain(self) -> list[str]:
        """
        Define the adsorbate elements that the estimator can handle.
        """
        pass

    @property
    @abstractmethod
    def surface_domain(self) -> list[str]:
        """
        Define the surface elements that the estimator can handle.
        """
        pass


    @abstractmethod
    def eval(self, inter: Intermediate, surf: Optional[Surface] = None) -> None:
        """
        Estimate the energy of a state.

        Args:
            inter (Intermediate): The intermediate.
            surf (Surface, optional): The surface. Defaults to None.
        """
        pass


class ReactionEnergyEstimator(ABC):
    """
    Base class for reaction properties estimators.
    """

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, reaction: ElementaryReaction) -> None:
        self.eval(reaction)

    @property
    @abstractmethod
    def adsorbate_domain(self) -> list[str]:
        """
        Define the adsorbate elements that the estimator can handle.
        """
        pass

    @property
    @abstractmethod
    def surface_domain(self) -> list[str]:
        """
        Define the surface elements that the estimator can handle.
        """
        pass

    @abstractmethod
    def eval(self, reaction: ElementaryReaction) -> None:
        """
        Estimate reaction properties.

        Args:
            reaction (ElementaryReaction): The reaction.
        """
        pass
