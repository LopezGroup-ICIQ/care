"""
Differential Plug-Flow Reactor (PFR) model.

Being it a zero-conversion model, conversion (X) is zero by definition, 
consequently yields (Y = X*S) are also zero. However, TOF and selectivity
can be computed, as well as apparent activation energy and reaction orders.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import isspmatrix_csr, csr_matrix
from time import time

from care.reactors.reactor import ReactorModel
from care.reactors.utils import net_rate, jacobian_fill_numba

import juliacall

jl = juliacall.newmodule("mkm")

jl.seval(
    """
    module SparsePFR
    using CUDA
    using DifferentialEquations
    using SparseArrays

    struct SparsePFRParams
        kd::Vector{Float64}
        kr::Vector{Float64}
        gas_mask::BitVector
        v_data::Vector{Int8}
        v_indices::Vector{Int}
        v_indptr::Vector{Int}
        vf_data::Vector{Int8}
        vf_indices::Vector{Int}
        vf_indptr::Vector{Int}
        vb_data::Vector{Int8}
        vb_indices::Vector{Int}
        vb_indptr::Vector{Int}
    end

    export SparsePFRParams, sparse_net_rate, ode_pfr!, sparse_jacobian

    function sparse_net_rate(u, p::SparsePFRParams)
        nr = length(p.kd)
        rates = zeros(Float64, nr)
        for r in 1:nr
            fprod = 1.0
            start_f = p.vf_indptr[r] + 1        # convert 0-based → 1-based
            stop_f  = p.vf_indptr[r+1]          # already exclusive in SciPy
            for idx in start_f:stop_f
                s = p.vf_indices[idx] + 1       # species index → 1-based
                exp = p.vf_data[idx]
                fprod *= u[s]^exp
            end

            bprod = 1.0
            start_b = p.vb_indptr[r] + 1
            stop_b  = p.vb_indptr[r+1]
            for idx in start_b:stop_b
                s = p.vb_indices[idx] + 1
                exp = p.vb_data[idx]
                bprod *= u[s]^exp
            end

            rates[r] = p.kd[r]*fprod - p.kr[r]*bprod
        end
        return rates
    end

    function ode_pfr!(du, u, p::SparsePFRParams, t::Float64)
        rates = sparse_net_rate(u, p)
        fill!(du, 0.0)
        for r in 1:length(rates)
            start_v = p.v_indptr[r]     # Python 0-based
            stop_v  = p.v_indptr[r+1]   # exclusive
            for idx in (start_v+1):stop_v   # Julia 1-based, inclusive
                s = p.v_indices[idx] + 1   # Python→Julia for species index
                du[s] += p.v_data[idx] * rates[r]
            end
        end
        for i in 1:length(p.gas_mask)
            if p.gas_mask[i] == 1
                du[i] = 0.0
            end
        end
    end

    function log_ode_pfr!(dx, x, p::SparsePFRParams, t::Float64)
        u = exp.(x)  # log-space -> physical space conversion
        du_physical = similar(u)
        ode_pfr!(du_physical, u, p, t)
        dx .= du_physical ./ u  # convert back to log-space with chain-rule
    end

    function sparse_jacobian!(J::SparseMatrixCSC{Float64,Int}, u, p::SparsePFRParams, t)
        # Compute out-of-place
        Jtmp = sparse_jacobian_outplace(u, p)

        # Clear J
        fill!(J.nzval, 0.0)

        # Copy values from Jtmp into J
        # This works if J has same sparsity structure as Jtmp
        for col in 1:size(Jtmp,2)
            for ptr in Jtmp.colptr[col]:(Jtmp.colptr[col+1]-1)
                row = Jtmp.rowval[ptr]
                val = Jtmp.nzval[ptr]
                J[row, col] = val
            end
        end
    end

    function sparse_jacobian_outplace(u, p::SparsePFRParams)
        nr = length(p.kd)
        ns = length(p.gas_mask)

        nnz_max = 0
        for r in 1:nr
            n_v  = p.v_indptr[r+1] - p.v_indptr[r]
            n_sf = p.vf_indptr[r+1] - p.vf_indptr[r]
            n_sb = p.vb_indptr[r+1] - p.vb_indptr[r]
            nnz_max += n_v * (n_sf + n_sb)
        end

        if nnz_max == 0
            return spzeros(Float64, ns, ns)
        end

        rows = Vector{Int}(undef, nnz_max)
        cols = Vector{Int}(undef, nnz_max)
        vals = Vector{Float64}(undef, nnz_max)
        pos = 1

        for r in 1:nr
            # forward/backward index ranges for reaction r
            sf_start = p.vf_indptr[r] + 1
            sf_stop  = p.vf_indptr[r+1]
            sb_start = p.vb_indptr[r] + 1
            sb_stop  = p.vb_indptr[r+1]

            # contributions from forward participants
            for idx in sf_start:sf_stop
                s = p.vf_indices[idx] + 1
                exp = Int(p.vf_data[idx])
                if exp == 0
                    continue
                end

                # product of forward terms excluding species s
                prod_except_s = 1.0
                for kdx in sf_start:sf_stop
                    j = p.vf_indices[kdx] + 1
                    ej = Int(p.vf_data[kdx])
                    if j == s
                        if ej - 1 > 0
                            prod_except_s *= u[j] ^ (ej - 1)
                        else
                            prod_except_s *= 1.0
                        end
                    else
                        prod_except_s *= u[j] ^ ej
                    end
                end
                dfr = p.kd[r] * exp * prod_except_s

                # distribute to stoichiometric rows (vT)
                vT_start = p.v_indptr[r] + 1
                vT_stop  = p.v_indptr[r+1]
                for jdx in vT_start:vT_stop
                    i = p.v_indices[jdx] + 1
                    if p.gas_mask[i] == 1
                        continue
                    end
                    coeff = Float64(p.v_data[jdx])
                    rows[pos] = i
                    cols[pos] = s
                    vals[pos] = coeff * dfr
                    pos += 1
                end
            end

            # contributions from backward participants
            for idx in sb_start:sb_stop
                s = p.vb_indices[idx] + 1
                exp = Int(p.vb_data[idx])
                if exp == 0
                    continue
                end

                # product of backward terms excluding species s
                prod_except_s = 1.0
                for kdx in sb_start:sb_stop
                    j = p.vb_indices[kdx] + 1
                    ej = Int(p.vb_data[kdx])
                    if j == s
                        if ej - 1 > 0
                            prod_except_s *= u[j] ^ (ej - 1)
                        else
                            prod_except_s *= 1.0
                        end
                    else
                        prod_except_s *= u[j] ^ ej
                    end
                end
                dbr = -p.kr[r] * exp * prod_except_s

                # distribute to stoichiometric rows (vT)
                vT_start = p.v_indptr[r] + 1
                vT_stop  = p.v_indptr[r+1]
                for jdx in vT_start:vT_stop
                    i = p.v_indices[jdx] + 1
                    if p.gas_mask[i] == 1
                        continue
                    end
                    coeff = Float64(p.v_data[jdx])
                    rows[pos] = i
                    cols[pos] = s
                    vals[pos] = coeff * dbr
                    pos += 1
                end
            end
        end

        if pos == 1
            return spzeros(Float64, ns, ns)
        else
            return sparse(rows[1:pos-1], cols[1:pos-1], vals[1:pos-1], ns, ns)
        end
    end
    end # module
    """
)

SparsePFR = jl.SparsePFR


class DifferentialPFR(ReactorModel):
    def __init__(
        self,
        v: np.ndarray = np.array([[]]),
        kd: np.ndarray = np.array([]),
        kr: np.ndarray = np.array([]),
        gas_mask: np.ndarray = np.array([]),
        inters: list[str] = None,
        pressure: float = 100000.0,
        temperature: float = 298.0,
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
        self.inters = inters or []  # List of intermediate species codes

        self.P = pressure  # Pressure of the reactor in Pascal
        self.T = temperature  # Temperature of the reactor in Kelvin

        self.sstol = None  # Tolerance for steady-state conditions
        self.sum_ddt, self.time = [], []

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
        sum_ddt = np.sum(abs(self.ode(t, y)))
        abssum_ddt_gas = np.sum(abs(self.ode(t, y)[self.gas_mask]))
        Py_gas = np.sum(y[self.gas_mask])
        print(
            f"Time: {t}    Sum_ddt: {sum_ddt}    Gas_ddt: {abssum_ddt_gas}    Gas_sum: {Py_gas}"
        )
        self.time.append(t)
        self.sum_ddt.append(sum_ddt)
        if sum_ddt <= self.sstol:
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
        gpu: bool = False,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
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
            gpu(bool): Flag to use GPU for the integration (only for Julia).
            analytical_jacobian(bool): Flag to use analytical Jacobian.
            impose_nonnegativity(bool): Flag to impose non-negativity on the solution.
            log_transform(bool): Flag to use log-transform on the concentrations. If set to True, 
                                    analytical_jacobian and impose_nonnegativity are ignored.
        Returns:
            (dict): Dictionary containing the solution of the ODE system.

        Notes:
            The integration is stopped when the sum of the absolute values of the derivatives reaches
            the steady-state tolerance 'sstol'.
        """
        if solver == "Julia":
            results = {}
            try:
                if gpu:
                    try:
                        time0 = time()
                        results["y"] = self.integrate_jl_gpu(
                            y0, rtol=rtol, atol=atol, sstol=sstol, tfin=tfin, 
                            analytical_jacobian=analytical_jacobian, 
                            impose_nonnegativity=impose_nonnegativity,
                            log_transform=log_transform
                        )
                        results["time"] = time() - time0
                        results["status"] = 1
                    except Exception as e:
                        print(f"Error: {e}")
                        print("Switching from GPU to CPU...")
                        time0 = time()
                        results["y"] = self.integrate_jl_cpu(
                            y0, rtol=rtol, atol=atol, sstol=sstol, tfin=tfin, 
                            analytical_jacobian=analytical_jacobian, 
                            impose_nonnegativity=impose_nonnegativity, 
                            log_transform=log_transform
                        )
                        results["time"] = time() - time0
                        results["status"] = 1
                else:
                    time0 = time()
                    results["y"] = np.array(self.integrate_jl_cpu(
                        y0, rtol=rtol, atol=atol, sstol=sstol, tfin=tfin, 
                        analytical_jacobian=analytical_jacobian, 
                        impose_nonnegativity=impose_nonnegativity,
                        log_transform=log_transform
                    ))
                    results["time"] = time() - time0
                    results["status"] = 1
            except Exception as e:
                print(f"Error: {e}")
                results["status"] = 0
        elif solver == "Python":
            self.sum_ddt = []
            self.sstol = sstol
            ode_events = (
                [self.steady_state, self.gas_change_event]
                if sstol
                else [self.gas_change_event]
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

            print(f"Integration time: {results['time']:.2f}s")

            results["y"] = results["y"][:, -1]
            results["time_ss"] = self.time
            results["sum_ddt"] = self.sum_ddt
        else:
            raise ValueError("Invalid solver. Choose between 'Python' or 'Julia'.")
        results["forward_rate"] = self.forward_rate(results["y"])
        results["backward_rate"] = self.backward_rate(results["y"])
        results["net_rate"] = self.net_rate(results["y"])
        results["consumption_rate"] = self.v_sparse.multiply(results["net_rate"])
        results["total_consumption_rate"] = results["consumption_rate"].sum(axis=1)
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
    
    def integrate_jl_cpu(
        self,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        sstol: float,
        tfin: float,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
    ) -> np.ndarray:
        """
        Integrate the ODE system using the Julia-based solver on CPU, sparse-aware.
        """
        v_transposed = self.v_sparse.T.tocsr()
        jl.p = SparsePFR.SparsePFRParams(
            self.kd, self.kr, self.gas_mask,
            v_transposed.data, v_transposed.indices, v_transposed.indptr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        jl.y0 = y0
        jl.log_transform = log_transform
        jl.analytical_jacobian = False if log_transform else analytical_jacobian
        jl.impose_nonnegativity = False if log_transform else impose_nonnegativity
        jl.atol, jl.rtol, jl.sstol, jl.tfin = atol, rtol, sstol, tfin

        jl.seval(
            """
            using DifferentialEquations, SparseArrays
            if analytical_jacobian
                J = spzeros(Float64, length(y0), length(y0))
                f = ODEFunction(SparsePFR.ode_pfr!, jac=(J,u,p,t)->SparsePFR.sparse_jacobian!(J,u,p,t), jac_prototype=J)
            else
                if log_transform
                    floor_value = 1e-70  # to avoid log(0)
                    y0 = log.(y0)
                    y0 = max.(y0, log(floor_value))
                    f = ODEFunction(SparsePFR.log_ode_pfr!)
                else
                    f = ODEFunction(SparsePFR.ode_pfr!)
                end
            end

            prob = ODEProblem(f, y0, (0, tfin), p)

            function condition(u, t, integrator)
                du = similar(u)
                if log_transform
                    SparsePFR.log_ode_pfr!(du, u, integrator.p, t)
                    du .= du .* exp.(u)  # convert back to physical space
                else
                    SparsePFR.ode_pfr!(du, u, integrator.p, t)
                end
                sum_abs_du = sum(abs.(du))
                println("$t: $sum_abs_du")
                return sum_abs_du <= sstol
            end

            function affect!(integrator)
                terminate!(integrator)
            end

            cb_steady_state = DiscreteCallback(condition, affect!)

            function nonnegativity_condition(u, t, integrator)
                true
            end

            function nonnegativity_affect!(integrator)
                integrator.u[integrator.u .< 0.0] .= 0.0
            end

            function element_condition(u, t, integrator)
                true
            end

            function element_affect!(integrator)
                terminate!(integrator)
            end

            cb_nonnegativity = DiscreteCallback(nonnegativity_condition, nonnegativity_affect!)

            if impose_nonnegativity
                cb_set = CallbackSet(cb_steady_state, cb_nonnegativity)
            else
                cb_set = CallbackSet(cb_steady_state)
            end

            sol = solve(prob, FBDF(autodiff=false), abstol=atol, reltol=rtol, callback=cb_set)
            
            if log_transform
                sol = exp.(Array(sol[end]))
            else
                sol = Array(sol[end])
            end
            """
        )

        return jl.sol
    
    def integrate_jl_gpu(
        self,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        sstol: float,
        tfin: float,
    ) -> np.ndarray:
        """
        Integrate the ODE system using the Julia-based solver on CPU, sparse-aware.
        """
        v_transposed = self.v_sparse.T.tocsr()
        jl.p = SparsePFR.SparsePFRParams(
            self.kd, self.kr, self.gas_mask,
            v_transposed.data, v_transposed.indices, v_transposed.indptr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        jl.y0 = y0
        jl.atol, jl.rtol, jl.sstol, jl.tfin = atol, rtol, sstol, tfin

        jl.seval(
            """
            using CUDA
            y0 = CuArray{Float64}(y0)
            CUDA.allowscalar(true)
            using DifferentialEquations
            f = ODEFunction(SparsePFR.ode_pfr!)
            prob = ODEProblem(f, y0, (0, tfin), p)

            function condition(u, t, integrator)
                du = similar(u)
                SparsePFR.ode_pfr!(du, u, integrator.p, t)  # sparse-aware
                sum_abs_du = sum(abs.(du))
                println("Condition check at time $t: $sum_abs_du")
                return sum_abs_du <= sstol
            end

            function affect!(integrator)
                terminate!(integrator)
            end
            cb_steady_state = DiscreteCallback(condition, affect!)
            function nonnegativity_condition(u, t, integrator)
                true
            end
            function nonnegativity_affect!(integrator)
                integrator.u[integrator.u .< 0.0] .= 0.0
            end
            cb_nonnegativity = DiscreteCallback(nonnegativity_condition, nonnegativity_affect!)

            # Combine the two callbacks into a single CallbackSet
            cb_set = CallbackSet(cb_steady_state, cb_nonnegativity)
            # sol = solve(prob, FBDF(autodiff=false), abstol=atol, reltol=rtol, callback=cb_set)
            sol = solve(prob, QNDF(autodiff=false), abstol=atol, reltol=rtol, callback=cb_set)
            sol = Array(sol[end])
            """
        )

        return jl.sol
