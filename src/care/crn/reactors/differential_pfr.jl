# DifferentialPFR implementation

module Differential_pfr

using CUDA, DifferentialEquations, DiffEqGPU

struct DifferentialPFR
    v_matrix::CuArray{Int8, 2}
    gas_mask::CuArray{Bool, 1}
    kd::CuArray{Float64, 1}
    kr::CuArray{Float64, 1}
    u0::CuArray{Float64, 1}
    v_forward::CuArray{Int8, 2}
    v_backward::CuArray{Int8, 2}
end

function ode(pfr::DifferentialPFR, t, y)
    net_rate = pfr.kd .* prod(y .^ pfr.v_forward, dims=2) .- pfr.kr .* prod(y .^ pfr.v_backward, dims=2)
    dydt = pfr.v_matrix * net_rate
    dydt[pfr.gas_mask] .= 0
    return dydt
end

function solve(pfr::DifferentialPFR, tspan)
    prob = ODEProblem((t, y) -> ode(pfr, t, y), pfr.u0, tspan)
    # Use stiff solver for better accuracy
    return solve(prob, Rosenbrock23())
end

function ode_pfr(y, p, t)
    net_rate = p.kd * prod(y .^ p.v_forward, dims=2) .- p.kr * prod(y .^ p.v_backward, dims=2)
    dydt = p.v * net_rate
    dydt[p.gas_mask] = 0
    return dydt
end
end