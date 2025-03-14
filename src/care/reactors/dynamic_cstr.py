"""
Dynamic Continuous Stirred Tank Reactor (CSTR) model.
"""

from math import pi
import numpy as np
from scipy.integrate import solve_ivp

from care.reactors.reactor import ReactorModel
from care.constants import N_AV, R


class DynamicCSTR(ReactorModel):
    def __init__(
        self,
        temperature: float = 298.0,
        pressure: float = 100000,
        v_matrix: np.ndarray = np.array([[]]),
        radius: float = 0.0254,
        length: float = 0.01,
        Q: float = 1e-6,
        m_cat: float = 1e-3,
        s_bet: float = 1e5,
        a_site: float = 1e-19,
    ):
        """
        Dynamic Continuous Stirred Tank Reactor (CSTR)
        Main assumptions:
            - Isothermal, isobaric
            - Dynamic behaviour to reach steady state (target) with ODE solver
            - Finite volume
            - Perfect mixing (zero transport phenomena)
        Args:
            temperature(float): Reactor temperature in K.
            pressure(float): Reactor pressure in Pa.
            v_matrix(ndarray): Stoichiometric matrix of the reaction network.
            radius(float): Radius of the tubular reactor [m]
            length(float): Axial length of the tubular reactor [m]
            Q(float): Inlet volumetric flowrate [m3 s-1]
            m_cat(float): catalyst mass [kg]
            s_bet(float): BET surface [m2 kg_cat-1]
            a_site(float): Active site area in m^2 active_site^-1
        """
        assert temperature > 0, "Temperature must be positive"
        assert pressure > 0, "Pressure must be positive"
        assert radius > 0, "Radius must be positive"
        assert length > 0, "Length must be positive"
        assert Q > 0, "Volumetric flowrate must be positive"
        assert m_cat > 0, "Catalyst mass must be positive"
        assert s_bet > 0, "BET surface must be positive"
        assert a_site > 0, "Active site area must be positive"

        self.temperature = temperature
        self.pressure = pressure
        self.v_matrix = v_matrix
        self.radius = radius
        self.length = length
        self.volume = (pi * radius**2) * length
        self.Q = Q
        self.tau = self.volume / self.Q  # Residence time
        self.m_cat = m_cat
        self.s_bet = s_bet
        self.a_site = a_site

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

        mat = np.zeros_like(self.v_matrix)
        mat[self.v_matrix < 0] = -self.v_matrix[self.v_matrix < 0]
        return mat.T

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
        mat = np.zeros_like(self.v_matrix)
        mat[self.v_matrix > 0] = self.v_matrix[self.v_matrix > 0]
        return mat.T

    def net_rate(self, y: np.ndarray, kd: np.ndarray, kr: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): Surface fractional coverage (unitless) ans partial pressure of gas reactants/products in Pa.
            kd, kr(ndarray): Kinetic constants of the direct/reverse steps.
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return kd * np.prod(y**self.stoic_forward, axis=1) - kr * np.prod(
            y**self.stoic_backward, axis=1
        )

    def ode(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        kr: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
        y0: np.ndarray,
    ) -> np.ndarray:
        dy = self.v_matrix @ self.net_rate(y, kd, kr)
        dy[gas_mask] *= (R * self.temperature * self.s_bet * self.m_cat) / (
            N_AV * self.volume * self.a_site
        )
        dy[gas_mask] += (y0[gas_mask] - y[gas_mask]) / self.tau
        return dy

    def jacobian(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        kr: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
        y0: np.ndarray,
    ) -> np.ndarray:
        # TODO Taken from Pymkm, readapt!
        J = np.zeros((len(y), len(y)))
        Jg = np.zeros((len(kd), len(y)))
        Jh = np.zeros((len(kd), len(y)))
        v_f = self.stoic_forward.copy()
        v_b = self.stoic_backward.copy()
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
                    Jh[r, s] = kr[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
                elif v_b[s, r] == 2:
                    v_b[s, r] -= 1
                    Jh[r, s] = 2 * kr[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
        J = self.v_matrix @ (Jg - Jh)
        J[gas_mask] *= (R * self.temperature * self.s_bet * self.m_cat) / (
            N_AV * self.volume * self.a_site
        )
        # for i in range(len(y) - NC_sur):
        #     J[NC_sur + i, NC_sur + i] -= 1 / self.tau  # TODO: check this
        return J

    def steady_state(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        kr: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
        y0: np.ndarray,
    ) -> float:
        sum_ddt = np.sum(abs(self.ode(t, y, kd, kr, gas_mask, sstol, y0)))
        print(t, sum_ddt)
        return 0 if sum_ddt <= sstol else sum_ddt

    steady_state.terminal = True

    def integrate(
        self,
        y0: np.ndarray,
        ode_params: tuple,
        rtol: float,
        atol: float,
        sstol: float,
        method: str = "BDF",
    ) -> dict:
        """
        Integrate the ODE system until steady-state conditions are reached.

        Args:
            y0(ndarray): Initial conditions for the ODE system.
            ode_params(tuple): Parameters for the ODE system, containing kdirect, kreverse, gas_mask
            rtol(float): Relative tolerance for the integration.
            atol(float): Absolute tolerance for the integration.
            ss_tol(float): Tolerance parameter for controlling automatic stop when
                            steady-state conditions are reached by the solver. When the sum of the
                            absolute values of the derivatives is below this value, the integration
                            is stopped.
            method(str): Integration method for the ODE system. Strongly recommended to use BDF or Radau
                            for stiff systems.

        Returns:
            (dict): Dictionary containing the solution of the ODE system.

        Notes:
            The integration is stopped when the sum of the absolute values of the derivatives is below
            the tolerance parameter `sstol`.
        """
        results = solve_ivp(
            self.ode,
            (0, 1e30),
            y0,
            method=method,
            events=self.steady_state,
            jac=None,  # self.jacobian to double check, to readapt
            args=ode_params + (sstol, y0),
            atol=atol,
            rtol=rtol,
        )
        results["rate"] = self.net_rate(
            results["y"][:, -1], ode_params[0], ode_params[1]
        )
        consumption_rate = np.zeros_like(self.v_matrix)
        for i in range(self.v_matrix.shape[0]):
            for j in range(self.v_matrix.shape[1]):
                consumption_rate[i, j] = self.v_matrix[i, j] * results["rate"][j]
        results["consumption_rate"] = consumption_rate
        performance = np.zeros((self.v_matrix.shape[0], 2))
        for i in range(self.v_matrix.shape[0]):
            if ode_params[2][i]:
                performance[i, 0] = self.conversion(i, y0, results["y"][:, -1])
                # performance[i, 1] = self.selectivity(i, gas_idxs[0][0], y0, results["y"][:, -1])
                # performance[i, 2] = self.yyield(i, gas_idxs[0][0], y0, results["y"][:, -1])
                performance[i, 1] = self.reaction_rate(y0, results["y"][:, -1], i)
        results["performance"] = performance
        return results

    def conversion(
        self, reactant_index: int, yin: np.ndarray, yout: np.ndarray
    ) -> float:
        """
        Conversion of reactant i.
        """
        return (
            1 - yout[reactant_index] / yin[reactant_index]
            if yin[reactant_index] > 0
            else 0
        )

    def selectivity(
        self, reactant_index: int, product_index: int, yin: np.ndarray, yout: np.ndarray
    ) -> float:
        """
        Selectivity of reactant i towards product j.
        """
        r = reactant_index
        p = product_index
        if yin[r] == 0:
            return 0
        else:
            return (yout[p] - yin[p]) / (yin[r] - yout[r])

    def yyield(
        self, reactant_index: int, product_index: int, yin: np.ndarray, yout: np.ndarray
    ) -> float:
        """
        Yield of reactant i towards product j.
        """
        X = self.conversion(reactant_index, yin, yout)
        S = self.selectivity(reactant_index, product_index, yin, yout)
        return X * S

    def reaction_rate(
        self, product_index: int, yin: np.ndarray, yout: np.ndarray
    ) -> float:
        """
        Reaction rate of product i in mol/s.
        """
        return (
            self.Q * (yout[product_index] - yin[product_index]) / (R * self.temperature)
        )
