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
        self.ode, self.jacobian, self.steady_state should have the same
        signature if scipy.solve_ivp is used to integrate the ODE system.
        """
        ...

    # @abstractmethod
    # def jacobian(self):
    #     """
    #     Provides the Jacobian matrix of the ODE system for the
    #     defined reactor model. J is a sparse square matrix.
    #     self.ode, self.jacobian, self.steady_state should have the same
    #     signature if scipy.solve_ivp is used to integrate the ODE system.
    #     """
    #     ...

    @abstractmethod
    def steady_state(self):
        """
        Defines the criteria needed to stop the integration. Typically,
        the termination occurs when steady-state conditions are reached.
        self.ode, self.jacobian, self.steady_state should have the same
        signature if scipy.solve_ivp is used to integrate the ODE system.
        """
        ...

    @abstractmethod
    def integrate(self):
        """
        Integrates the ODE system until steady-state conditions are reached.
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

    def units_of_measure(self, key=None):
        units = {'pressure': 'Pa', 
                 'temperature': 'K', 
                 'kinetic_constants': 's^-1}', 
                 'potential': 'V', 
                 'pH': 'unitless', 
                 'catalyst_mass': 'kg', 
                 'volumetric_flowrate': 'm^3 s^-1', 
                 'length': 'm', 
                 'conversion': '%', 
                 'yield': '%', 
                 'selectivity': '%'}
        if key == None:
            return units
        else:
            if key in units.keys():
                return units[key]
            else:
                return units
