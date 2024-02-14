import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from care.crn.reactors.reactor import ReactorModel
from care.crn.microkinetic import net_rate

class DifferentialPFR(ReactorModel):
    def __init__(self,  
                 v: np.ndarray, 
                 kd: np.ndarray,
                 kr: np.ndarray, 
                 gas_mask: np.ndarray):
        """
        Differential Plug-Flow Reactor (PFR)
        Main assumptions of the reactor model:
            - Isothermal, isobaric
            - Steady-state conditions
            - Finite volume
            - Perfect mixing (zero transport phenomena)

        Args:
            v(ndarray): Stoichiometric matrix of the system.
            kd(ndarray): Kinetic constants of the direct steps.
            kr(ndarray): Kinetic constants of the reverse steps.
            gas_mask(ndarray): Boolean array indicating which species are in the gas phase.
        """

        self.v_dense = v
        self.v_forward_dense = np.zeros_like(self.v_dense, dtype=np.int8)
        self.v_forward_dense[self.v_dense < 0] = -self.v_dense[self.v_dense < 0]
        self.v_forward_dense = self.v_forward_dense.T
        self.v_backward_dense = np.zeros_like(self.v_dense, dtype=np.int8)
        self.v_backward_dense[self.v_dense > 0] = self.v_dense[self.v_dense > 0]
        self.v_backward_dense = self.v_backward_dense.T

        self.v_sparse = csr_matrix(v, dtype=np.int8)
        self.v_forward_sparse = csr_matrix(self.v_forward_dense, dtype=np.int8)
        self.v_backward_sparse = csr_matrix(self.v_backward_dense, dtype=np.int8)

        self.nr = self.v_dense.shape[1]  # number of reactions
        self.nc = self.v_dense.shape[0]  # number of species

        self.kd = kd  # Direct kinetic constants
        self.kr = kr  # Reverse kinetic constants

        self.gas_mask = gas_mask

        self.sstol = 1e-10  # Tolerance for steady-state conditions

    def __str__(self) -> str:
        return f"Differential Plug-Flow Reactor (PFR) with {self.nr} elementary reactions and {self.nc} species."

    def net_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return self.kd * np.prod(y**self.v_forward_dense, axis=1) - self.kr * np.prod(
            y**self.v_backward_dense, axis=1
        )

    def ode(
        self,
        t: float,
        y: np.ndarray,
    ) -> np.ndarray:
        dydt = self.v_sparse.dot(net_rate(y, self.kd, self.kr, self.v_forward_dense, self.v_backward_dense))
        dydt[self.gas_mask] = 0
        return dydt

    def jacobian(
        self,
        t: float,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian matrix of DifferentialPFR model.
        Considers elementary reactions with stoichiometric coefficients of 1 or 2.
        Not really helpful as its computation is very expensive.
        """
        Jg = np.zeros((self.nr, self.nc), dtype=np.float64)
        Jh = np.zeros((self.nr, self.nc), dtype=np.float64)
        vf = self.v_forward_dense.T.copy()
        vb = self.v_backward_dense.T.copy()
        for r in range(self.nr):
            for s in range(self.nc):
                if vf[s, r] == 1:
                    vf[s, r] -= 1
                    Jg[r, s] = self.kd[r] * np.prod(y ** vf[:, r])
                    vf[s, r] += 1
                elif vf[s, r] == 2:
                    vf[s, r] -= 1
                    Jg[r, s] = 2 * self.kd[r] * np.prod(y ** vf[:, r])
                    vf[s, r] += 1
                if vb[s, r] == 1:
                    vb[s, r] -= 1
                    Jh[r, s] = self.kr[r] * np.prod(y ** vb[:, r])
                    vb[s, r] += 1
                elif vb[s, r] == 2:
                    vb[s, r] -= 1
                    Jh[r, s] = 2 * self.kr[r] * np.prod(y ** vb[:, r])
                    vb[s, r] += 1
        J = self.v_dense @ (Jg - Jh)
        J[self.gas_mask, :] = 0
        return J
    
    @property    
    def jac_sparsity(self) -> np.ndarray:
        """
        Matrix defining where the jacobian is non-zero.
        Not useful as the jacobian is not used in the integration.
        """
        J = np.zeros((self.nc, self.nc), dtype=np.float64)
        Jg = np.zeros((self.nr, self.nc), dtype=np.float64)
        Jh = np.zeros((self.nr, self.nc), dtype=np.float64)
        for r in range(self.nr):
            for s in range(self.nc):
                if self.v_forward_dense.T[s, r] != 0:
                    Jg[r, s] = 1
                if self.v_backward_dense.T[s, r] != 0:                    
                    Jh[r, s] = 1
        J = self.v_dense @ (Jg - Jh)
        J[self.gas_mask, :] = 0
        return J


    def steady_state(
        self,
        t: float,
        y: np.ndarray,
    ) -> float:
        sum_ddt = np.sum(abs(self.ode(t, y)))
        print(f"Sum of the absolute derivatives: {sum_ddt}")
        return 0 if sum_ddt <= self.sstol else 1

    steady_state.terminal = True
    steady_state.direction = 0

    def integrate(
        self,
        y0: np.ndarray,
        solver: str = "Julia",
        rtol: float = 1e-10,
        atol: float = 1e-32,
        sstol: float = 1e-10,
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
        if solver == "Julia":
            results = {}
            results["y"] = self.integrate_jl(y0)
        else:
            self.sstol = sstol
            results = solve_ivp(self.ode,
                                (0, 1e30),  
                                y0,
                                method="BDF",
                                events=self.steady_state,
                                jac=None,
                                atol=atol,
                                rtol=rtol, 
                                jac_sparsity=None)
            results["y"] = results["y"][:, -1]
        results["rate"] = self.net_rate(results["y"])
        consumption_rate = np.zeros_like(self.v_dense, dtype=np.float64)
        for i in range(self.v_dense.shape[0]):
            for j in range(self.v_dense.shape[1]):
                    consumption_rate[i, j] = self.v_dense[i, j] * results["rate"][j]
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
    
    def integrate_jl(self, y0):
        """
        Integrate the ODE system using the Julia-based solver.
        """
        from julia.api import Julia
        jl = Julia()
        from julia import Main
        Main.y0 = y0
        Main.v = self.v_dense
        Main.vf = self.v_forward_dense
        Main.vb = self.v_backward_dense
        Main.kd, Main.kr = self.kd, self.kr
        Main.gas_mask = self.gas_mask
        Main.eval("""
        using CUDA
        using DifferentialEquations
        using SparseArrays
        CUDA.allowscalar(true)
        y0 = CuArray{Float64}(y0)
        v = CuArray{Int8}(sparse(v))
        kd = CuArray{Float64}(kd)
        kr = CuArray{Float64}(kr)
        vf = CuArray{Int8}(sparse(vf))
        vb = CuArray{Int8}(sparse(vb))
        gas_mask = CuArray{Bool}(gas_mask)
        p = (v = v, kd = kd, kr = kr, gas_mask = gas_mask, vf =  vf, vb = vb)
        # println("y0: $(sizeof(y0)/1000) v: $(sizeof(v)/1000) kd: $(sizeof(kd)/1000) kr: $(sizeof(kr)/1000) vf: $(sizeof(vf)/1000) vb: $(sizeof(vb)/1000) gas_mask: $(sizeof(gas_mask)/1000)")
        # println("y0: $(size(y0)) v: $(size(v)) kd: $(size(kd)) kr: $(size(kr)) vf: $(size(vf)) vb: $(size(vb)) gas_mask: $(size(gas_mask))")
        """)
        Main.eval("""
        function ode_pfr!(du, u, p, t)
            net_rate = p.kd .* prod((u .^ p.vf')', dims=2) .- p.kr .* prod((u .^ p.vb')', dims=2)
            du .= p.v * net_rate
            du[p.gas_mask] .= 0.0
            println(sum(abs.(du)))
        end
        """)
        Main.eval("""
        prob = SteadyStateProblem(ode_pfr!, y0, p)
        """)
        Main.eval("""
        sol = solve(prob, DynamicSS(KenCarp4()))
        """)
        Main.eval("sol = Array(sol)")
        Main.eval("""
        CUDA.allowscalar(false)
        """)
        solution = Main.sol
        return solution
