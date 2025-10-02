module SparsePFR
    using DifferentialEquations
    using SparseArrays
    using Printf
    using LinearSolve

    struct SparsePFRParams{T}
        kd::Vector{T}
        kr::Vector{T}
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

    function sparse_net_rate(u, p::SparsePFRParams{T}) where T
        nr = length(p.kd)
        rates = zeros(T, nr)
        for r in 1:nr
            fprod = one(T)
            start_f = p.vf_indptr[r] + 1      # convert 0-based → 1-based
            stop_f  = p.vf_indptr[r+1]        # already exclusive in SciPy
            for idx in start_f:stop_f
                s = p.vf_indices[idx] + 1   # species index → 1-based
                exp = p.vf_data[idx]
                fprod *= u[s]^exp
            end

            bprod = one(T)
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

    function ode_pfr!(du::AbstractVector{T}, u::AbstractVector{T}, p::SparsePFRParams{T}, t) where T
        rates = sparse_net_rate(u, p)
        fill!(du, zero(eltype(du)))
        for r in 1:length(rates)
            start_v = p.v_indptr[r]      # Python 0-based
            stop_v  = p.v_indptr[r+1]    # exclusive
            for idx in (start_v+1):stop_v    # Julia 1-based, inclusive
                s = p.v_indices[idx] + 1   # Python→Julia for species index
                du[s] += p.v_data[idx] * rates[r]
            end
        end
        for i in 1:length(p.gas_mask)
            if p.gas_mask[i] == 1
                du[i] = zero(T)
            end
        end
    end

    function log_ode_pfr!(dx, x, p::SparsePFRParams, t)
        u = exp.(x)  # log-space -> physical space conversion
        du_physical = similar(u)
        ode_pfr!(du_physical, u, p, t)
        # Add a floor to prevent division by zero for species that go to zero
        u[u .== 0] .= eps(eltype(u))
        dx .= du_physical ./ u  # convert back to log-space with chain-rule
    end

    function sparse_jacobian!(J::SparseMatrixCSC{T,Int}, u, p::SparsePFRParams{T}, t) where T
        Jtmp = sparse_jacobian_outplace(u, p)
        # This is a more robust way to update the jacobian prototype
        J.nzval .= zero(T)
        rows, cols, vals = findnz(Jtmp)
        for (r, c, v) in zip(rows, cols, vals)
            J[r, c] = v
        end
    end

    function sparse_jacobian_outplace(u, p::SparsePFRParams{T}) where T
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
            return spzeros(T, ns, ns)
        end

        rows = Vector{Int}(undef, nnz_max)
        cols = Vector{Int}(undef, nnz_max)
        vals = Vector{T}(undef, nnz_max)
        pos = 1

        for r in 1:nr
            sf_start = p.vf_indptr[r] + 1
            sf_stop  = p.vf_indptr[r+1]
            sb_start = p.vb_indptr[r] + 1
            sb_stop  = p.vb_indptr[r+1]

            for idx in sf_start:sf_stop
                s = p.vf_indices[idx] + 1
                exp = Int(p.vf_data[idx])
                if exp == 0
                    continue
                end

                prod_except_s = one(T)
                for kdx in sf_start:sf_stop
                    j = p.vf_indices[kdx] + 1
                    ej = Int(p.vf_data[kdx])
                    if j == s
                        if ej - 1 > 0
                            prod_except_s *= u[j] ^ (ej - 1)
                        end
                    else
                        prod_except_s *= u[j] ^ ej
                    end
                end
                dfr = p.kd[r] * exp * prod_except_s

                vT_start = p.v_indptr[r] + 1
                vT_stop  = p.v_indptr[r+1]
                for jdx in vT_start:vT_stop
                    i = p.v_indices[jdx] + 1
                    if p.gas_mask[i] == 1
                        continue
                    end
                    coeff = T(p.v_data[jdx])
                    rows[pos] = i
                    cols[pos] = s
                    vals[pos] = coeff * dfr
                    pos += 1
                end
            end

            for idx in sb_start:sb_stop
                s = p.vb_indices[idx] + 1
                exp = Int(p.vb_data[idx])
                if exp == 0
                    continue
                end

                prod_except_s = one(T)
                for kdx in sb_start:sb_stop
                    j = p.vb_indices[kdx] + 1
                    ej = Int(p.vb_data[kdx])
                    if j == s
                        if ej - 1 > 0
                            prod_except_s *= u[j] ^ (ej - 1)
                        end
                    else
                        prod_except_s *= u[j] ^ ej
                    end
                end
                dbr = -p.kr[r] * exp * prod_except_s

                vT_start = p.v_indptr[r] + 1
                vT_stop  = p.v_indptr[r+1]
                for jdx in vT_start:vT_stop
                    i = p.v_indices[jdx] + 1
                    if p.gas_mask[i] == 1
                        continue
                    end
                    coeff = T(p.v_data[jdx])
                    rows[pos] = i
                    cols[pos] = s
                    vals[pos] = coeff * dbr
                    pos += 1
                end
            end
        end

        if pos == 1
            return spzeros(T, ns, ns)
        else
            return sparse(rows[1:pos-1], cols[1:pos-1], vals[1:pos-1], ns, ns)
        end
    end


    """
    This is the main entry point function from Python.
    It takes all the input data, sets up the ODE problem, solves it, and returns the result.
    """
    function setup_and_solve(
        y0_in, kd_in, kr_in, gas_mask,
        v_data, v_indices, v_indptr,
        vf_data, vf_indices, vf_indptr,
        vb_data, vb_indices, vb_indptr,
        atol, rtol, sstol, tfin,
        analytical_jacobian, impose_nonnegativity, log_transform, precision
    )
        # --- 1. Determine Numeric Type and Convert Data ---
        T = (precision > 64) ? BigFloat : Float64
        if T == BigFloat
            setprecision(BigFloat, precision)
            # When using BigFloat, data from Python comes as strings to preserve precision
            y0 = parse.(BigFloat, y0_in)
            kd = parse.(BigFloat, kd_in)
            kr = parse.(BigFloat, kr_in)
        else
            y0 = Vector{Float64}(y0_in)
            kd = Vector{Float64}(kd_in)
            kr = Vector{Float64}(kr_in)
        end

        p = SparsePFRParams{T}(
            kd, kr, gas_mask,
            v_data, v_indices, v_indptr,
            vf_data, vf_indices, vf_indptr,
            vb_data, vb_indices, vb_indptr
        )
        J_proto = nothing
        if T == Float64 && analytical_jacobian
            J_proto = sparse_jacobian_outplace(y0, p)
        end

        # --- 3. Setup ODE Function and Problem ---
        if log_transform
            floor_value = T(1e-70) # Set a floor to avoid log(0)
            y0_transformed = log.(max.(y0, floor_value))
            f = ODEFunction(log_ode_pfr!)
        else
            y0_transformed = y0
            if analytical_jacobian
                J_proto = sparse_jacobian_outplace(y0, p) # Create a prototype
                f = ODEFunction(ode_pfr!, jac=(J,u,p,t)->sparse_jacobian!(J,u,p,t), jac_prototype=J_proto)
            else
                f = ODEFunction(ode_pfr!)
            end
        end

        tspan = (zero(T), T(tfin))
        prob = ODEProblem(f, y0_transformed, tspan, p)

        # --- Define Callbacks ---
        function condition(u, t, integrator)
            du = similar(u)
            if log_transform
                u_phys = exp.(u)
                ode_pfr!(du, u_phys, integrator.p, t)
            else
                ode_pfr!(du, u, integrator.p, t)
            end
            sum_abs_du = sum(abs.(du))
            @printf("%s: %s\n", t, sum_abs_du)
            return sum_abs_du <= sstol
        end

        affect!(integrator) = terminate!(integrator)
        cb_steady_state = DiscreteCallback(condition, affect!)

        cb_nonnegativity = let
            nonnegativity_condition(u, t, integrator) = true
            function nonnegativity_affect!(integrator)
                u_type = eltype(integrator.u)
                integrator.u[integrator.u .< zero(u_type)] .= zero(u_type)
            end
            DiscreteCallback(nonnegativity_condition, nonnegativity_affect!)
        end

        # Don't apply non-negativity constraint in log-space
        cb_set = (impose_nonnegativity && !log_transform) ?
                 CallbackSet(cb_steady_state, cb_nonnegativity) :
                 CallbackSet(cb_steady_state)

        # --- 5. Solve the Problem ---
        solver = if T == BigFloat
            linsolve = KrylovJL_GMRES()
            Rodas5(autodiff=false, linsolve=linsolve)  # TODO: consider KenCarp4, TRBDF2, Rodas5P, RadauIIA5
        else
            FBDF(autodiff=false)
        end
        sol = solve(prob, solver, abstol=atol, reltol=rtol, callback=cb_set, save_everystep=false)

        # --- 6. Return Final State ---
        final_state = if log_transform
            exp.(sol.u[end])
        else
            sol.u[end]
        end

        return Array(final_state) # Convert back to a standard Array for Python
    end

end 
