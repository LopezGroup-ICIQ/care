"""Module providing the interface for the reactor models in pymkm."""

from abc import ABC, abstractmethod
from math import pi

import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

from care.constants import N_AV, R
from care.crn.microkinetic import net_rate, stoic_backward, stoic_forward

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


class DifferentialPFR(ReactorModel):
    def __init__(self, 
                 temperature: float, 
                 pressure: float,
                 v_matrix: np.ndarray, 
                 ss_tol: float = 1e-6):
        """
        Differential Plug-Flow Reactor (PFR)
        Main assumptions of the reactor model:
            - Isothermal, isobaric
            - Steady-state conditions
            - Finite volume
            - Perfect mixing

        Args:
            temperature(float): Temperature of the differential reactor [K]
            pressure(float): Pressure of the differential reactor [Pa]
            ss_tol(float): Tolerance parameter for controlling automatic stop when
                           steady-state conditions are reached by the solver.
        """
        self.temperature = temperature
        self.pressure = pressure
        self.v_matrix = v_matrix
        self.reactor_type = "Differential PFR"
        self.ss_tol = ss_tol
        self.ODE_params = {
            "reltol": 1e-12,
            "abstol": 1e-64,
            "tfin": 1e3}


    @property
    def stoic_forward(self) -> np.ndarray:
        """
        Filter function for the stoichiometric matrix.
        Negative elements are considered and changed of sign in order to
        compute the direct reaction rates.
        Args:
            matrix(ndarray): Stoichiometric matrix
        Returns:
            mat(ndarray): Filtered matrix for constructing forward reaction rates.
        """
        mat = np.zeros([self.v_matrix.shape[0], self.v_matrix.shape[1]])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if self.v_matrix[i][j] < 0:
                    mat[i][j] = -self.v_matrix[i][j]
        return mat

    @property
    def stoic_backward(self) -> np.ndarray:
        """
        Filter function for the stoichiometric matrix.
        Positive elements are considered and kept in order to compute
        the reverse reaction rates.
        Args:
            matrix(ndarray): stoichiometric matrix
        Returns:
            mat(ndarray): Filtered matrix for constructing reverse reaction rates.
        """
        mat = np.zeros([self.v_matrix.shape[0], self.v_matrix.shape[1]])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if self.v_matrix[i][j] > 0:
                    mat[i][j] = self.v_matrix[i][j]
        return mat
    
    # @jit(nopython=True)
    def net_rate(self, y: np.ndarray, 
             kd: np.ndarray, 
             ki: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
            kd, kr(ndarray): kinetic constants of the direct/reverse steps.
            v_matrix(ndarray): stoichiometric matrix of the system.
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return kd * np.prod(y**self.stoic_forward.T, axis=1) - ki * np.prod(y**self.stoic_backward.T, axis=1)

    # @jit(nopython=True)
    def ode(self, _, y, kd, ki, gas_mask: np.ndarray) -> np.ndarray:
        
        dy = self.v_matrix @ self.net_rate(y, kd, ki)  # Surface species        
        dy[gas_mask] = 0  # Mask gas species
        print(_)
        return dy

    def jacobian(self, time, y, kd, ki, v_matrix, NC_sur) -> np.ndarray:
        # TODO Taken from Pymkm, readapt!
        J = np.zeros((len(y), len(y)))
        Jg = np.zeros((len(kd), len(y)))
        Jh = np.zeros((len(kd), len(y)))
        v_f = stoic_forward(v_matrix)
        v_b = stoic_backward(v_matrix)
        for r in range(len(kd)):
            for s in range(len(y)):
                if v_f[s, r] == 1:
                    v_f[s, r] -= 1
                    Jg[r, s] = kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                elif v_f[s, r] == 2:
                    v_f[s, r] -= 1
                    Jg[r, s] = 2 * kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                if v_b[s, r] == 1:
                    v_b[s, r] -= 1
                    Jh[r, s] = ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
                elif v_b[s, r] == 2:
                    v_b[s, r] -= 1
                    Jh[r, s] = 2 * ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
        J = v_matrix @ (Jg - Jh)
        J[NC_sur:, :] = 0.0
        return J

    def steady_state(self, time, y, kd, ki, gas_mask) -> float:
        error = np.sum(abs(self.ode(time, y, kd, ki, gas_mask)))
        criteria = 0 if error <= self.ss_tol else error
        return criteria

    steady_state.terminal = True

    def integrate(self,
        y_0: np.ndarray,
        ode_params: tuple,
    ) -> dict:
        """
        Integrate the ODE system until steady-state conditions are reached.
        Args:
            y_0(ndarray): Initial conditions for the ODE system.
            ode_params(tuple): Parameters for the ODE system, containing kdirect, kreverse, gas_mask.

        Returns:
            (dict): Dictionary containing the solution of the ODE system.
        """
        return solve_ivp(
            self.ode,
            (0, 1e6),
            y_0,
            method="BDF", 
            events=self.steady_state,
            jac=None,  # To readapt
            args=ode_params,
            atol=self.ODE_params['abstol'],
            rtol=self.ODE_params['reltol'])


    def conversion(self, P_in, P_out) -> float:
        return 1 - (P_out / P_in)

    def selectivity(self, r_target, r_tot) -> float:
        return r_target / r_tot

    def reaction_rate(self):
        pass

    def yyield(self, P_in, P_out, r_target, r_tot) -> float:
        return self.conversion(P_in, P_out) * self.selectivity(r_target, r_tot)


class DynamicCSTR(ReactorModel):
    def __init__(
        self,
        temperature: float,
        pressure: float,
        radius: float=0.01,
        length: float=0.01,
        Q: float=1e-6,
        m_cat: float=1e-3,
        s_bet: float=1e5,
        a_site: float=1e-19,
        ss_tol: float=1e-5,
    ):
        """
        Dynamic Continuous Stirred Tank Reactor (CSTR)
        Main assumptions:
            - Isothermal, isobaric
            - Dynamic behaviour to reach steady state (target) with ODE solver
            - Finite volume
            - Perfect mixing
        Args:
            temperature(float): Temperature of the tubular reactor [K]
            pressure(float): Pressure of the tubular reactor [Pa]
            radius(float): Radius of the tubular reactor [m]
            length(float): Axial length of the tubular reactor [m]
            Q(float): Inlet volumetric flowrate [m3 s-1]
            m_cat(float): catalyst mass [kg]
            s_bet(float): BET surface [m2 kg_cat-1]
            a_site(float): Active site area [m2/ active site]
            ss_tol(float): Tolerance parameter for controlling automatic stop when
                           steady-state conditions are reached by the solver.
        """
        self.reactor_type = "Dynamic CSTR"
        self.temperature = temperature
        self.pressure = pressure
        self.radius = radius
        self.length = length
        self.volume = (pi * radius**2) * length
        self.Q = Q
        self.tau = self.volume / self.Q
        self.m_cat = m_cat
        self.s_bet = s_bet
        self.a_site = a_site
        self.ss_tol = ss_tol

    def ode(self, time, y, kd, ki, v_matrix, NC_sur, P_in):
        # Surface species
        dy = v_matrix @ net_rate(y, kd, ki, v_matrix)
        # Gas species
        dy[NC_sur:] *= R * self.temperature / (N_AV * self.volume)
        dy[NC_sur:] *= self.s_bet * self.m_cat / self.a_site
        dy[NC_sur:] += (P_in - y[NC_sur:]) / self.tau
        return dy

    def jacobian(self, time, y, kd, ki, v_matrix, NC_sur, temperature, P_in):
        # TODO Taken from Pymkm, readapt!
        J = np.zeros((len(y), len(y)))
        Jg = np.zeros((len(kd), len(y)))
        Jh = np.zeros((len(kd), len(y)))
        v_f = stoic_forward(v_matrix)
        v_b = stoic_backward(v_matrix)
        for r in range(len(kd)):
            for s in range(len(y)):
                if v_f[s, r] == 1:
                    v_f[s, r] -= 1
                    Jg[r, s] = kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                elif v_f[s, r] == 2:
                    v_f[s, r] -= 1
                    Jg[r, s] = 2 * kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                if v_b[s, r] == 1:
                    v_b[s, r] -= 1
                    Jh[r, s] = ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
                elif v_b[s, r] == 2:
                    v_b[s, r] -= 1
                    Jh[r, s] = 2 * ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
        J = v_matrix @ (Jg - Jh)
        J[NC_sur:, :] *= (R * temperature * self.s_bet * self.m_cat) / (
            N_AV * self.volume * self.a_site
        )
        for i in range(len(y) - NC_sur):
            J[NC_sur + i, NC_sur + i] -= 1 / self.tau
        return J

    def termination_event(self, time, y, kd, ki, v_matrix, NC_sur, temperature, P_in):
        error = np.sum(
            abs(self.ode(time, y, kd, ki, v_matrix, NC_sur, temperature, P_in))
        )
        criteria = 0 if error <= self.ss_tol else error
        return criteria

    termination_event.terminal = True

    def conversion(self, gas_reactant_index, P_in, P_out):
        """
        Returns the conversion of reactant i
        Internal function used for the dynamic CSTR model
        """
        index = gas_reactant_index
        X = 1 - (P_out[index] / P_in[index])
        return X

    def selectivity(self, gas_reactant_index, gas_product_index, P_in, P_out):
        """
        Returns the selectivity of reactant i to product j
        Internal function used for the dynamic CSTR model
        """
        r = gas_reactant_index
        p = gas_product_index
        S = (P_out[p] - P_in[p]) / (P_in[r] - P_out[r])
        return S

    def yyield(self, gas_reactant_index, gas_product_index, P_in, P_out):
        """
        Returns the yield of reactant i to product j.
        Internal function used for the dynamic CSTR model
        """
        X = self.conversion(gas_reactant_index, P_in, P_out)
        S = self.selectivity(gas_reactant_index, gas_product_index, P_in, P_out)
        return X * S

    def reaction_rate(self, P_in, P_out, temperature):
        return self.Q * (P_out - P_in) / (R * temperature)
