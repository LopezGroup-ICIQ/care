"""
Differential Plug-Flow Reactor (PFR) model.

Being it a zero-conversion model, conversion (X) is zero by definition, 
consequently yields (Y = X*S) are also zero. However, TOF and selectivity
can be computed, as well as apparent activation energy and reaction orders.
"""
import os 

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import isspmatrix_csr, csr_matrix
from time import time

from care.constants import INTER_ELEMS
from care.reactors.reactor import ReactorModel
from care.reactors.utils import net_rate, jacobian_fill_numba

import juliacall

jl = juliacall.newmodule("mkm")
script_dir = os.path.dirname(os.path.abspath(__file__))
julia_solver_path = os.path.join(script_dir, "pfr_solver.jl")
jl.seval(f'include("{julia_solver_path}")')
SparsePFR = jl.SparsePFR


class DifferentialPFR(ReactorModel):
    def __init__(
        self,
        v: np.ndarray = np.array([[]]),
        kd: np.ndarray = np.array([]),
        kr: np.ndarray = np.array([]),
        gas_mask: np.ndarray = np.array([]),
        inters: dict = None,
        pressure: float = 100000.0,
        temperature: float = 298.0,
        print_progress: bool = True,
    ):
        """
        Differential Plug-Flow Reactor (PFR)
        Main assumptions of the reactor model:
            - Isothermal, isobaric
            - Steady-state conditions
            - Finite volume
            - Perfect mixing (zero transport phenomena)

        Args:
            v(np.ndarray): Stoichiometric matrix of the system.
            kd(np.ndarray): Kinetic constants of the direct steps.
            kr(np.ndarray): Kinetic constants of the reverse steps.
            gas_mask(np.ndarray): Boolean array indicating which species are in the gas phase.
            inters(list): List of intermediate species codes.
            pressure(float): Pressure of the reactor in Pascal.
            temperature(float): Temperature of the reactor in Kelvin.
            print_progress(bool): Flag to print the progress of the ODE integration.
                                  If set to True, for each step the code will print 
                                  the time in seconds and the sum of the absolute values of the derivatives.
        """
        if not isspmatrix_csr(v):
            raise ValueError("Stoichiometric matrix v must be a scipy sparse CSR matrix")
        self.v_sparse = v
        self.v_forward_sparse = self.v_sparse.multiply(self.v_sparse < 0).multiply(-1).T.tocsr()
        self.v_backward_sparse = self.v_sparse.multiply(self.v_sparse > 0).T.tocsr()

        self.sparsity = (1 - self.v_sparse.nnz / (self.v_sparse.shape[0] * self.v_sparse.shape[1])) * 100

        self.nr = self.v_sparse.shape[1]  # number of reactions
        self.nc = self.v_sparse.shape[0]  # number of species

        self.kd = kd  # Forward kinetic constants
        self.kr = kr  # Backward kinetic constants

        self.gas_mask = gas_mask  # Boolean array indicating which species are in the gas phase
        self.inters_info = inters
        self.inters = inters["codes"] or []  # List of intermediate species codes
        self.elements = inters["elements"]

        self.P = pressure  # Pressure of the reactor in Pascal
        self.T = temperature  # Temperature of the reactor in Kelvin

        self.sstol = 0.01  # Tolerance for steady-state conditions
        self.sum_ddt, self.time = [], []
        self.print_progress = print_progress

    def __str__(self) -> str:
        y = f"Differential Plug-Flow Reactor (PFR) with {self.nr} elementary reactions and {self.nc} species\n"
        y += f"Pressure: {self.P} Pa, Temperature: {self.T} K\n"
        return y
    
    def forward_rate(self, y: np.ndarray) -> np.ndarray:
        rates = np.empty(self.nr, dtype=np.float64)
        v = self.v_forward_sparse.tocsr()

        for j in range(v.shape[0]):
            start, end = v.indptr[j], v.indptr[j+1]
            idx = v.indices[start:end]
            exps = v.data[start:end]
            rates[j] = self.kd[j] * np.prod(y[idx] ** exps)
        return rates

    def backward_rate(self, y: np.ndarray) -> np.ndarray:
        rates = np.empty(self.nr, dtype=np.float64)
        v = self.v_backward_sparse.tocsr()

        for j in range(v.shape[0]):
            start, end = v.indptr[j], v.indptr[j+1]
            idx = v.indices[start:end]
            exps = v.data[start:end]
            rates[j] = self.kr[j] * np.prod(y[idx] ** exps)

        return rates

    def net_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return self.forward_rate(y) - self.backward_rate(y)

    def ode(
        self,
        _: float,
        y: np.ndarray,
    ) -> np.ndarray:
        rates = net_rate(
            y,
            self.kd, self.kr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        dydt = self.v_sparse.dot(rates)
        dydt[self.gas_mask] = 0.0
        return dydt

    def jacobian(self, _, y: np.ndarray) -> csr_matrix:
        """
        Assemble sparse Jacobian as CSR. Returns scipy.sparse.csr_matrix (nc x nc).
        """
        # ensure reaction-row CSR (shape: n_reactions x n_species)
        vT = self.v_sparse.T.tocsr()

        # arrays for forward/backward (already reaction-row in your __init__)
        sf_data, sf_indices, sf_indptr = (
            self.v_forward_sparse.data,
            self.v_forward_sparse.indices,
            self.v_forward_sparse.indptr,
        )
        sb_data, sb_indices, sb_indptr = (
            self.v_backward_sparse.data,
            self.v_backward_sparse.indices,
            self.v_backward_sparse.indptr,
        )

        # vT arrays
        vT_data, vT_indices, vT_indptr = vT.data, vT.indices, vT.indptr

        # precompute conservative upper bound for triplets:
        nnz_max = 0
        for r in range(self.nr):
            n_v = vT_indptr[r+1] - vT_indptr[r]     # stoichiometry nonzeros for reaction r
            n_sf = sf_indptr[r+1] - sf_indptr[r]   # forward participants
            n_sb = sb_indptr[r+1] - sb_indptr[r]   # backward participants
            nnz_max += n_v * (n_sf + n_sb)

        if nnz_max == 0:
            # empty Jacobian (no reactions)
            return csr_matrix((self.nc, self.nc), dtype=np.float64)

        # preallocate triplet arrays
        rows = np.empty(nnz_max, dtype=np.int32)
        cols = np.empty(nnz_max, dtype=np.int32)
        vals = np.empty(nnz_max, dtype=np.float64)

        used = jacobian_fill_numba(
            y, self.kd, self.kr,
            sf_data, sf_indices, sf_indptr,
            sb_data, sb_indices, sb_indptr,
            vT_data, vT_indices, vT_indptr,
            rows, cols, vals,
        )

        if used == 0:
            J = csr_matrix((self.nc, self.nc), dtype=np.float64)
        else:
            # slice to used entries and build CSR (duplicates will be summed)
            J = csr_matrix((vals[:used], (rows[:used], cols[:used])), shape=(self.nc, self.nc))

        J = J.tolil()
        J[self.gas_mask, :] = 0.0
        J = J.tocsr()

        return J

    def steady_state(
        self,
        t: float,
        y: np.ndarray,
    ) -> float:
        """Steady state termination condition.
        It triggers when the sum of coverages is 1, and the elemental
        input and output flows are equal (in=out for C, H, etc.)
        """
        in_div_out = {"*": sum(y[self.gas_mask])}
        rates = net_rate(
            y,
            self.kd, self.kr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        dydt = self.v_sparse.dot(rates)
        inflow, outflow = {}, {}
        elem_dict = {k: v for k, v in self.inters_info.items() if k in INTER_ELEMS}
        for elem, counts in elem_dict.items():
            inflow[elem] = 0.0
            outflow[elem] = 0.0
            for i, coeff in enumerate(counts):
                if self.gas_mask[i]:
                    contrib = coeff * dydt[i]
                    if contrib > 0:
                        outflow[elem] += contrib
                    elif contrib < 0:
                        inflow[elem] += abs(contrib)
            if inflow[elem] == 0.0 and outflow[elem] == 0.0:
                in_div_out[elem] = 1
            else:
                in_div_out[elem] = inflow[elem] / (outflow[elem] + np.finfo(float).eps)
        self.time.append(t)
        sum_balances = sum(in_div_out.values())
        self.sum_ddt.append(sum_balances)
        if self.print_progress:
            print(
                f"t={t}s    sum_balances = {sum_balances}"
            )
        max_dev = max([abs(x - 1) for x in in_div_out.values()])
        if max_dev <= self.sstol:
            print("STEADY-STATE  REACHED!!!")
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
        tfin: float,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
        precision: int = 64,
        jl_solver: str = "FBDF", 
        maxiters: int = 100_000,
        show_progress: bool = False,
        **kwargs,
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
            analytical_jacobian(bool): Flag to use analytical Jacobian.
            impose_nonnegativity(bool): Flag to impose non-negativity on the solution.
            log_transform(bool): Flag to use log-transform on the concentrations. If set to True, 
                                    analytical_jacobian and impose_nonnegativity are ignored.
            precision(int): Precision for the Julia solver. Default to 64 (double precision).
            jl_solver(str): ODE solver in DifferentialEquations.jl, considered only with double precision.
                            Default to FBDF.
            maxiters(int): Maximum number of ODE steps. Defaults to 100_000
            show_progress(bool): If True, the maximum value among the elemental balances IN/OUT is shown.
                                Convergence occurs when IN/OUT flows for all elements (C,H,O, ...) and the sum
                                of the surface coverages are between
                                0.99 and 1.01. Integration wtop when the printed number reaches 0.01.
        Returns:
            (dict): Dictionary containing the solution of the ODE system.

        Notes:
            The integration is stopped when the sum of the absolute values of the derivatives reaches
            the steady-state tolerance 'sstol'.
        """
        if solver == "Julia":
            results = {}
            time0 = time()
            y, t = self.integrate_jl_cpu(
                y0, rtol=rtol, atol=atol, tfin=tfin,
                analytical_jacobian=analytical_jacobian, 
                impose_nonnegativity=impose_nonnegativity,
                log_transform=log_transform, 
                precision=precision, 
                jl_solver=jl_solver,
                maxiters=maxiters,
                show_progress=show_progress
            )
            results["y"] = y
            results["t"] = t
            results["time"] = time() - time0
            results["status"] = 1
        elif solver == "Python":
            self.sum_ddt = []
            ode_events = (
                [self.steady_state, self.gas_change_event]
            )
            time0 = time()
            results = solve_ivp(
                self.ode,
                (0, tfin),
                y0,
                method="BDF",
                events=ode_events,
                jac=self.jacobian if analytical_jacobian else None,
                atol=atol,
                rtol=rtol,
                jac_sparsity=None,
            )
            results["time"] = time() - time0
            results["y"] = results["y"][:, -1]
            results["t"] = results["t"][-1]
            results["time_ss"] = self.time
            results["sum_ddt"] = self.sum_ddt
        else:
            raise ValueError("Invalid solver. Choose between 'Python' or 'Julia'.")
        results["forward_rate"] = self.forward_rate(results["y"])
        results["backward_rate"] = self.backward_rate(results["y"])
        results["reversibility"] = results["forward_rate"] / results["backward_rate"]
        results["net_rate"] = self.net_rate(results["y"])
        results["consumption_rate"] = self.v_sparse.multiply(results["net_rate"])
        results["total_consumption_rate"] = results["consumption_rate"].sum(axis=1)
        results["gas_mask"] = self.gas_mask
        results["inters"] = self.inters
        results["inters_info"] = self.inters_info
        results["y0"] = y0
        results["T"] = self.T
        results["P"] = self.P
        results["rtol"] = rtol
        results["atol"] = atol
        results["tfin"] = tfin
        results["v"] = self.v_sparse
        results["solver"] = solver
        results["jl_solver"] = jl_solver if solver == "Julia" else "Python"
        results["precision"] = precision if solver == "Julia" else 64
        results["maxiters"] = maxiters if solver == "Julia" else None
        reactants_idxs = np.where((self.gas_mask) & (y0 > 0))[0]
        products_idxs = np.where((self.gas_mask) & (y0 == 0))[0]
        results["reactants_idxs"] = reactants_idxs
        results["products_idxs"] = products_idxs
        conversion_vector = np.zeros(len(reactants_idxs))
        results["conversion"] = conversion_vector
        selectivity_matrix = np.zeros((len(reactants_idxs), len(products_idxs), len(self.elements)))
        yield_matrix = np.zeros((len(reactants_idxs), len(products_idxs), len(self.elements)))
        r = results["total_consumption_rate"]
        n = self.inters_info
        for e, elem in enumerate(self.elements):
            for i, reactant in enumerate(reactants_idxs):
                n_elem_reactant = n[elem][reactant]
                for j, product in enumerate(products_idxs):
                    if n_elem_reactant == 0:
                        selectivity_matrix[i, j, e] = np.nan  # example: selectivity of H2 to CO makes no sense, thus nan
                    else:
                        selectivity_matrix[i, j, e] = r[product] * n[elem][product] / (abs(r[reactant]) * n_elem_reactant)
                    yield_matrix[i, j, e] = 0.0
        results["selectivity"] = {elem: selectivity_matrix[:, :, e] for e, elem in enumerate(self.elements)}
        results["yield"] = {elem: yield_matrix[:, :, e] for e, elem in enumerate(self.elements)}
        return results

    def conversion(self, reactant_idx: int, y: np.ndarray) -> float:
        """
        Conversion of the reactant i.
        By definition, conversion is 0 due to infinitesimal volume of the reactor.
        """
        return 0.0

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
    
    def integrate_jl_cpu(
        self,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        tfin: float,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
        precision: int = 64,
        jl_solver: str = "FBDF",
        maxiters: int = 1000000,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Integrate the ODE system using the Julia-based solver on CPU, supporting Float64 or BigFloat.
        """

        if precision > 64:
            y0_in = [str(x) for x in y0]
            kd_in = [str(x) for x in self.kd]
            kr_in = [str(x) for x in self.kr]
        else:
            y0_in = y0
            kd_in = self.kd
            kr_in = self.kr
            
        elem_dict = {k: v for k, v in self.inters_info.items() if k in INTER_ELEMS}
        jl_elem_dict = jl.Dict([(k, jl.Vector(v)) for k, v in elem_dict.items()])

        vT = self.v_sparse.T.tocsr()
        solution, time = jl.SparsePFR.setup_and_solve(
            y0_in, kd_in, kr_in, self.gas_mask,
            vT.data.astype(np.int8), vT.indices, vT.indptr,
            self.v_forward_sparse.data.astype(np.int8), self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data.astype(np.int8), self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
            atol, rtol, self.sstol, tfin,
            analytical_jacobian, impose_nonnegativity, log_transform, precision, jl_solver, maxiters, show_progress, 
            jl_elem_dict
        )
        dtype = np.float64 if precision == 64 else np.float128
        return np.array(solution, dtype=dtype), float(time)
