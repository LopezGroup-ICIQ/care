import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from care.crn.reactors.reactor import ReactorModel
from care.crn.microkinetic import net_rate, net_rate_cuda

class DifferentialPFR(ReactorModel):
    def __init__(self, 
                 temperature: float, 
                 pressure: float, 
                 v_matrix: np.ndarray):
        """
        Differential Plug-Flow Reactor (PFR)
        Main assumptions of the reactor model:
            - Isothermal, isobaric
            - Steady-state conditions
            - Finite volume
            - Perfect mixing (zero transport phenomena)

        Args:
            temperature(float): Reactor temperature in K.
            pressure(float): Reactor pressure in Pa.
            v_matrix(ndarray): Stoichiometric matrix of the reaction network.
        """
        assert temperature > 0, "Temperature must be positive"
        assert pressure > 0, "Pressure must be positive"

        self.temperature = temperature
        self.pressure = pressure
        self.v_matrix_dense = v_matrix
        self.v_matrix_sparse = csr_matrix(v_matrix, dtype=np.int8)

        self.stoic_forward_dense = np.zeros_like(self.v_matrix_dense, dtype=np.int8)
        self.stoic_forward_dense[self.v_matrix_dense < 0] = -self.v_matrix_dense[self.v_matrix_dense < 0]
        self.stoic_forward_dense = self.stoic_forward_dense.T
        self.stoic_forward_sparse = csr_matrix(self.stoic_forward_dense)

        self.stoic_backward_dense = np.zeros_like(self.v_matrix_dense, dtype=np.int8)
        self.stoic_backward_dense[self.v_matrix_dense > 0] = self.v_matrix_dense[self.v_matrix_dense > 0]
        self.stoic_backward_dense = self.stoic_backward_dense.T
        self.stoic_backward_sparse = csr_matrix(self.stoic_backward_dense)

        self.nr = self.v_matrix_dense.shape[1]  # number of reactions
        self.nc = self.v_matrix_dense.shape[0]  # number of species

    def net_rate(self, y: np.ndarray, kd: np.ndarray, kr: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
            kd, kr(ndarray): kinetic constants of the direct/reverse steps.
            v_matrix(ndarray): stoichiometric matrix of the system.
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return kd * np.prod(y**self.stoic_forward_dense, axis=1) - kr * np.prod(
            y**self.stoic_backward_dense, axis=1
        )

    def ode(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        ki: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
    ) -> np.ndarray:
        # dy = self.v_matrix_sparse.dot(self.net_rate(y, kd, ki))
        dy = self.v_matrix_sparse.dot(net_rate(y, kd, ki, self.stoic_forward_dense, self.stoic_backward_dense))
        dy[gas_mask] = 0
        return dy

    def jacobian(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        ki: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
    ) -> np.ndarray:
        """
        Jacobian matrix of DifferentialPFR model.
        Considers elementary reactions with stoichiometric coefficients of 1 or 2.
        Not really helpful as its computation is very expensive.
        """
        J = np.zeros((len(y), len(y)), dtype=np.float64)
        Jg = np.zeros((len(kd), len(y)), dtype=np.float64)
        Jh = np.zeros((len(kd), len(y)), dtype=np.float64)
        vf = self.stoic_forward_dense.T.copy()
        vb = self.stoic_backward_dense.T.copy()
        for r in range(len(kd)):
            for s in range(len(y)):
                if vf[s, r] == 1:
                    vf[s, r] -= 1
                    Jg[r, s] = kd[r] * np.prod(y ** vf[:, r])
                    vf[s, r] += 1
                elif vf[s, r] == 2:
                    vf[s, r] -= 1
                    Jg[r, s] = 2 * kd[r] * np.prod(y ** vf[:, r])
                    vf[s, r] += 1
                if vb[s, r] == 1:
                    vb[s, r] -= 1
                    Jh[r, s] = ki[r] * np.prod(y ** vb[:, r])
                    vb[s, r] += 1
                elif vb[s, r] == 2:
                    vb[s, r] -= 1
                    Jh[r, s] = 2 * ki[r] * np.prod(y ** vb[:, r])
                    vb[s, r] += 1
        J = self.v_matrix_dense @ (Jg - Jh)
        J[gas_mask, :] = 0
        return J
    
    def jac_sparsity(self,
                     gas_mask: np.ndarray) -> np.ndarray:
        """
        Matrix defining where the jacobian is non-zero.
        Not useful as the jacobian is not used in the integration.
        """
        J = np.zeros((self.nc, self.nc), dtype=np.float64)
        Jg = np.zeros((self.nr, self.nc), dtype=np.float64)
        Jh = np.zeros((self.nr, self.nc), dtype=np.float64)
        for r in range(self.nr):
            for s in range(self.nc):
                if self.stoic_forward_dense.T[s, r] != 0:
                    Jg[r, s] = 1
                if self.stoic_backward_dense.T[s, r] != 0:                    
                    Jh[r, s] = 1
        J = self.v_matrix_dense @ (Jg - Jh)
        J[gas_mask, :] = 0
        return J


    def steady_state(
        self,
        t: float,
        y: np.ndarray,
        kd: np.ndarray,
        ki: np.ndarray,
        gas_mask: np.ndarray,
        sstol: float,
    ) -> float:
        sum_ddt = np.sum(abs(self.ode(t, y, kd, ki, gas_mask, sstol)))
        print(f"Sum of the derivatives: {sum_ddt}")
        return 0 if sum_ddt <= sstol else sum_ddt

    steady_state.terminal = True
    steady_state.direction = 0

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
        results = solve_ivp(self.ode,
                            (0, 1e30),  # high value to allow reaching steady state
                            y0,
                            method=method,
                            events=self.steady_state,
                            jac=None,  # jacobian to to readapt
                            args=ode_params + (sstol,),
                            atol=atol,
                            rtol=rtol, 
                            jac_sparsity=None)
        results["rate"] = self.net_rate(results["y"][:, -1], ode_params[0],
                                         ode_params[1])
        consumption_rate = np.zeros_like(self.v_matrix_dense, dtype=np.float64)
        for i in range(self.v_matrix_dense.shape[0]):
            for j in range(self.v_matrix_dense.shape[1]):
                    consumption_rate[i, j] = self.v_matrix_dense[i, j] * results["rate"][j]
        results["consumption_rate"] = consumption_rate
        return results

    def conversion(self, 
                   reactant_idx: int, 
                   y: np.ndarray) -> float:
        """
        Conversion of the reactant i.
        By definition, conversion is 0 due to infinitesimal volume of the reactor.
        """
        return 1 - y[reactant_idx, -1] / y[reactant_idx, 0]

    def selectivity(self, 
                    target_idx: int, 
                    product_idxs: list[int], 
                    consumption_rate: np.ndarray) -> float:
        """
        Selectivity towards a target product.
        As conversion is zero, the selectivity is computed as the ratio between the
        consumption rate of the target product and the total consumption rate.
        Args:
            target_idx(int): Index of the target product.
            product_idxs(list[int]): Indexes of the products. It must contain the target index.
            consumption_rate(ndarray): Consumption rate matrix of each species.

        Returns:
            (float): Selectivity towards the target product (between 0 and 1)
        """
        r_target = np.sum(consumption_rate[target_idx, :])
        r_tot = np.sum(consumption_rate[product_idxs, :])
        return r_target / r_tot

    def reaction_rate(self, 
                      product_idx: int, 
                      consumption_rate: np.ndarray) -> float:
        return np.sum(consumption_rate[product_idx, :])

    def yyield(self, 
               reactant_idx: int, 
               target_idx: int,
               product_idxs: list[int], 
               consumption_rate:np.ndarray) -> float:
        """
        Yield of reactant i towards product j.
        By definition, yield is 0 due to infinitesimal volume of the reactor.
        """
        X = self.conversion(reactant_idx)
        S = self.selectivity(target_idx, product_idxs, consumption_rate)
        return X * S