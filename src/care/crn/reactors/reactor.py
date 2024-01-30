"""Module providing the interface for the reactor models in pymkm."""

from abc import ABC, abstractmethod


class ReactorModel(ABC):
    def __init__(self):
        """
        Abstract class for the implementation of reactor models.
        """
        pass

    @abstractmethod
    def ode(self):
        """
        Provides the ODE system representing the reactor model,
        based on species, material, momentum and energy balances.
        """
        ...

    @abstractmethod
    def jacobian(self):
        """
        Provides the Jacobian matrix of the ODE system for the
        defined reactor model. J is a square matrix, typically sparse.
        """
        ...

    @abstractmethod
    def steady_state(self):
        """
        Defines the criteria needed to stop the integration. Typically,
        the termination occurs when steady-state conditions are reached.
        """
        ...

    @abstractmethod
    def conversion(self):
        """
        Provides the conversion of reactant i.
        """
        ...

    @abstractmethod
    def selectivity(self):
        """
        Provides the selectivity of reactant i towards product j.
        """
        ...

    @abstractmethod
    def reaction_rate(self):
        """
        Provides the production rate of the specific species.
        """
        ...
