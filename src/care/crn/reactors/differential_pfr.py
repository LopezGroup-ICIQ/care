import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from time import time

from care.crn.reactors.reactor import ReactorModel
from care.crn.microkinetic import net_rate


class DifferentialPFR(ReactorModel):
    def __init__(
        self,
        v: np.ndarray,
        kd: np.ndarray,
        kr: np.ndarray,
        gas_mask: np.ndarray,
        inters: list[str],
        pressure: float,
        temperature: float,
    ):
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
        # Get sparsity of the stoichiometric matrix
        self.sparsity = (1 - (np.count_nonzero(v) / (v.shape[0] * v.shape[1]))) * 100
        print(f"Sparsity of the stoichiometric matrix: {self.sparsity:.2f}%")
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
        self.inters = inters

        self.P = pressure  # Pressure of the reactor
        self.T = temperature  # Temperature of the reactor

        self.sstol = None  # Tolerance for steady-state conditions
        self.sum_ddt, self.time = [], []

    def __str__(self) -> str:
        y = f"Differential Plug-Flow Reactor (PFR) with {self.nr} elementary reactions and {self.nc} species\n"
        y += f"Pressure: {self.P} Pa, Temperature: {self.T} K\n"

        return y

    def forward_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the forward reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
        Returns:
            (ndarray): forward reaction rate of the elementary reactions [1/s].
        """
        return self.kd * np.prod(y**self.v_forward_dense, axis=1)

    def backward_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the backward reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
        Returns:
            (ndarray): backward reaction rate of the elementary reactions [1/s].
        """
        return self.kr * np.prod(y**self.v_backward_dense, axis=1)

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
        _: float,
        y: np.ndarray,
    ) -> np.ndarray:
        dydt = self.v_sparse.dot(
            net_rate(y, self.kd, self.kr, self.v_forward_dense, self.v_backward_dense)
        )
        dydt[self.gas_mask] = 0.0
        # print(_)
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
                    Jg[r, s] = 10
                if self.v_backward_dense.T[s, r] != 0:
                    Jh[r, s] = 17
        J = self.v_dense @ (Jg - Jh)
        J[self.gas_mask, :] = 0
        return J

    def steady_state(
        self,
        t: float,
        y: np.ndarray,
    ) -> float:
        sum_ddt = np.sum(abs(self.ode(t, y)))
        abssum_ddt_gas = np.sum(abs(self.ode(t, y)[self.gas_mask]))
        Py_gas = np.sum(y[self.gas_mask])
        print(
            f"Time: {t}    Sum_ddt: {sum_ddt}    Gas_ddt: {abssum_ddt_gas}    Gas_sum: {Py_gas}"
        )
        self.time.append(t)
        self.sum_ddt.append(sum_ddt)
        if sum_ddt <= self.sstol:
            print("STEADY-STATE CONDITIONS REACHED!")
            return 0
        return 1

    steady_state.terminal = True
    steady_state.direction = 0

    def gas_change_event(
        self,
        _: float,
        y: np.ndarray,
    ) -> float:
        """
        Event function to detect when the gas phase changes.
        """
        Py_gas = np.sum(y[self.gas_mask])
        return 0 if Py_gas != self.P else 1

    gas_change_event.terminal = False
    gas_change_event.direction = 0

    def integrate(
        self,
        y0: np.ndarray,
        solver: str,
        rtol: float,
        atol: float,
        sstol: float,
        tfin: float,
    ) -> dict:
        """
        Integrate the ODE system up to steady-state.

        Args:
            y0(ndarray): Initial conditions for the ODE system.
            solver(str): Solver to use for the integration. Options are 'Python' or 'Julia'.
            rtol(float): Relative tolerance for the integration.
            atol(float): Absolute tolerance for the integration.
            sstol(float): Tolerance for steady-state conditions.
            tfin(float): Final time for the integration.

        Returns:
            (dict): Dictionary containing the solution of the ODE system.

        Notes:
            The integration is stopped when the sum of the absolute values of the derivatives reaches
            the steady-state tolerance 'sstol'.
        """

        TFIN = tfin if tfin else 1e6

        if solver == "Julia":
            results = {}
            try:
                results["y"] = self.integrate_jl(
                    y0, rtol=rtol, atol=atol, sstol=sstol, tfin=TFIN
                )
                results["status"] = 1
            except Exception as e:
                print(f"Error: {e}")
                results["status"] = 0
        elif solver == "Python":
            self.sum_ddt = []
            self.sstol = sstol
            time0 = time()
            ode_events = (
                [self.steady_state, self.gas_change_event]
                if sstol
                else [self.gas_change_event]
            )
            results = solve_ivp(
                self.ode,
                (0, TFIN),
                y0,
                method="BDF",
                events=ode_events,
                jac=None,
                atol=atol,
                rtol=rtol,
                jac_sparsity=None,
            )
            results["time"] = time() - time0

            print(f"Integration time: {results['time']:.2f}s")

            results["y"] = results["y"][:, -1]
            results["time_ss"] = self.time
            results["sum_ddt"] = self.sum_ddt
        else:
            raise ValueError("Invalid solver. Choose between 'Python' or 'Julia'.")
        results["forward_rate"] = self.forward_rate(results["y"])
        results["backward_rate"] = self.backward_rate(results["y"])
        results["net_rate"] = self.net_rate(results["y"])
        consumption_rate = np.zeros_like(self.v_dense, dtype=np.float64)
        for i in range(self.v_dense.shape[0]):
            for j in range(self.v_dense.shape[1]):
                consumption_rate[i, j] = self.v_dense[i, j] * results["net_rate"][j]
        results["consumption_rate"] = consumption_rate
        results["total_consumption_rate"] = np.sum(consumption_rate, axis=1)
        return results

    def conversion(self, reactant_idx: int, y: np.ndarray) -> float:
        """
        Conversion of the reactant i.
        By definition, conversion is 0 due to infinitesimal volume of the reactor.
        """
        return 1 - y[reactant_idx, -1] / y[reactant_idx, 0]

    def selectivity(
        self, target_idx: int, product_idxs: list[int], consumption_rate: np.ndarray
    ) -> float:
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

    def reaction_rate(self, product_idx: int, consumption_rate: np.ndarray) -> float:
        return np.sum(consumption_rate[product_idx, :])

    def yyield(
        self,
        reactant_idx: int,
        target_idx: int,
        product_idxs: list[int],
        consumption_rate: np.ndarray,
    ) -> float:
        """
        Yield of reactant i towards product j.
        By definition, yield is 0 due to infinitesimal volume of the reactor.

        Note:
            the method is called yyield to avoid conflicts with the yield keyword in Python.
        """
        X = self.conversion(reactant_idx)
        S = self.selectivity(target_idx, product_idxs, consumption_rate)
        return X * S

    def integrate_jl(
        self,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        sstol: float,
        tfin: float,
        gpu: bool = False,
    ) -> np.ndarray:
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
        Main.atol = atol
        Main.rtol = rtol
        Main.sstol = sstol
        Main.J_sparsity = self.jac_sparsity
        Main.tfin = tfin
        Main.gpu = gpu
        Main.eval(
            """
        using DifferentialEquations
        # GPU section
        if gpu
            using CUDA, DiffEqGPU
            y0 = CuArray{Float64}(y0)
            v = CuArray{Int8}(v)
            vf = CuArray{Int8}(vf)
            vb = CuArray{Int8}(vb)
            kd = CuArray{Float64}(kd)
            kr = CuArray{Float64}(kr)
            gas_mask = CuArray{Bool}(gas_mask)
        end
        p = (v = v, kd = kd, kr = kr, gas_mask = gas_mask, vf =  vf, vb = vb)
        """
        )
        Main.eval(
            """
        function ode_pfr!(du, u, p, t)
            net_rate = p.kd .* prod((u .^ p.vf')', dims=2) .- p.kr .* prod((u .^ p.vb')', dims=2)
            du .= p.v * net_rate
            du[p.gas_mask] .= 0.0
            println(t,"    ", sum(abs.(du)))
        end
        """
        )
        Main.eval(
            """
        f = ODEFunction(ode_pfr!)
        """
        )
        Main.eval(
            """
        prob = ODEProblem(f, y0, (0, tfin), p)
        """
        )
        Main.eval(
            """
        function condition(u, t, integrator)
            du = similar(u)
            ode_pfr!(du, u, integrator.p, t)
            sum_abs_du = sum(abs.(du))  # Calculate the absolute sum of du
            return sum_abs_du <= sstol
        end

        function affect!(integrator)
            println("STEADY-STATE CONDITIONS REACHED!")
            terminate!(integrator)
        end
        cb = DiscreteCallback(condition, affect!)
        """
        )
        Main.eval(
            """
        sol = solve(prob, FBDF(), abstol=atol, reltol=rtol, callback=cb)
        """
        )
        Main.eval("sol = Array(sol[end])")
        return Main.sol
