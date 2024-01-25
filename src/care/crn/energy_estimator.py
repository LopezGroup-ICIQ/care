"""
Base class for energy estimators.
"""

from typing import Optional
from abc import ABC, abstractmethod

from care import Intermediate, Surface

class EnergyEstimator(ABC):
    """
    Base class for intermediate energy estimators.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def estimate_energy(self, 
                        inter: Intermediate, 
                        surf: Optional[Surface] = None) -> None:
        """
        Estimate the energy of a state.

        Args:
            inter (Intermediate): The intermediate.
            surf (Surface, optional): The surface. Defaults to None.
        """
        pass