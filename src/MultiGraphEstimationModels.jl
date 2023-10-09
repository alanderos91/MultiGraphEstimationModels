module MultiGraphEstimationModels

using LinearAlgebra, Statistics, Distributions, SpecialFunctions
using PoissonRandom, Random, StableRNGs
using Polyester

import Base: show

#
#   EDGE DISTRIBUTIONS
#

abstract type AbstractEdgeDistribution end

abstract type NBParam end

struct MeanScale <: NBParam end
struct MeanDispersion <: NBParam end

struct PoissonEdges <: AbstractEdgeDistribution end

struct NegBinEdges{param<:NBParam} <: AbstractEdgeDistribution end

NegBinEdges(param::T) where T <: NBParam = NegBinEdges{T}()
NegBinEdges() = NegBinEdges(MeanScale())

export PoissonEdges, NBParam, MeanScale, MeanDispersion, NegBinEdges

#
#   UTILITIES
#
include("utilities.jl")

#
#   MODEL INTERFACE
#

abstract type AbstractMultiGraphModel{distT,intT,floatT,covT} end

function (::Type{M})(dist::AbstractEdgeDistribution, observed) where M <: AbstractMultiGraphModel
    M(dist, observed, nothing)
end

function (::Type{M})(dist::AbstractEdgeDistribution, observed, params::NamedTuple) where M <: AbstractMultiGraphModel
    M(dist, observed, nothing, params)
end

function (::Type{M})(dist::PoissonEdges, observed, covariates) where M <: AbstractMultiGraphModel
    params = (scale=Inf, dispersion=0.0)
    M(dist, observed, covariates, params)
end

function (::Type{M})(dist::NegBinEdges, observed, covariates) where M <: AbstractMultiGraphModel
    r = Inf
    params = (scale=r, dispersion=1/r)
    M(dist, observed, covariates, params)
end

function update_expectations!(model::AbstractMultiGraphModel)
    update_expectations!(model, model.covariate)
end

function __allocate_buffers__(model::AbstractMultiGraphModel{distT}) where distT
    __allocate_buffers__(distT(), model)
end

include(joinpath("models", "UndirectedModel.jl"))
include(joinpath("models", "DirectedModel.jl"))

export UndirectedMultiGraphModel, DirectedMultiGraphModel

#
#   SIMULATION
#

function default_propensity(rng, nnodes)
    rand(rng, Uniform(0.0, 4.0), nnodes)
end

function simulate_with_pois!(rng, observed, expected, propensity)
    # this assumes propensity[i] > 0 for every node i
    any(isequal(0), propensity) && error("Cannot simulate multigraph with disconnected node")
    nnodes = length(propensity)
    for j in 1:nnodes-1
        node_degree = 0
        while node_degree == 0
            for i in j+1:nnodes
                mu_ij = propensity[i] * propensity[j]
                observed[i,j] = observed[j,i] = pois_rand(rng, mu_ij)
                expected[i,j] = expected[j,i] = mu_ij
                node_degree += observed[i,j]
            end
        end
    end
end

function simulate_with_nbin!(rng, observed, expected, propensity, r)
    # this assumes propensity[i] > 0 for every node i
    any(isequal(0), propensity) && error("Cannot simulate multigraph with disconnected node")
    nnodes = length(propensity)
    for j in 1:nnodes-1
        node_degree = 0
        while node_degree == 0
            for i in j+1:nnodes
                mu_ij = propensity[i] * propensity[j]
                pi_ij = mu_ij / (mu_ij + r)
                D = NegativeBinomial(r, 1-pi_ij)
                observed[i,j] = observed[j,i] = rand(rng, D)
                expected[i,j] = expected[j,i] = mu_ij
                node_degree += observed[i,j]
            end
        end
    end
end

function simulate_propensity_model(::PoissonEdges, nnodes::Int;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    propensity::AbstractVector=default_propensity(rng, nnodes),
)
    # sanity check
    length(propensity) != nnodes && error("Number of nodes must match number of propensities.")
    any(<=(0), propensity) && error("Propensities must be positive.")

    # simulate Poisson expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    simulate_with_pois!(rng, observed, expected, propensity)

    example = UndirectedMultiGraphModel(PoissonEdges(), observed)
    copyto!(example.propensity, propensity)
    copyto!(example.expected, expected)

    return example
end

function simulate_propensity_model(dist::NegBinEdges, nnodes::Int;
    dispersion::Real=1.0,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    propensity::AbstractVector=default_propensity(rng, nnodes),
)
    # sanity check
    length(propensity) != nnodes && error("Number of nodes must match number of propensities.")
    any(<=(0), propensity) && error("Propensities must be positive.")

    # set scale parameter, r = 1/dispersion
    r = 1 / dispersion

    # simulate Negative Binomial expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    simulate_with_nbin!(rng, observed, expected, propensity, r)

    params = (scale=r, dispersion=dispersion)
    example = UndirectedMultiGraphModel(dist, observed, nothing, params)
    copyto!(example.propensity, propensity)
    copyto!(example.expected, expected)

    return example
end

function simulate_covariate_model(::PoissonEdges, nnodes::Int, ncovar::Int;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    propensity::AbstractVector=default_propensity(rng, nnodes),
)
    # sanity check
    length(propensity) != nnodes && error("Number of nodes must match number of propensities.")
    any(<=(0), propensity) && error("Propensities must be positive.")

    # simulate p covariates for each of the m nodes; iid N(0,1)
    design_matrix = randn(rng, ncovar, nnodes)

    # center and scale
    μ = mean(design_matrix, dims=2)
    σ = std(design_matrix, dims=2)
    covariate = (design_matrix .- μ) ./ σ

    # approximately generate coefficients that produce the target propensity values
    response = log.(propensity)
    coefficient = covariate' \ (log(10, nnodes) * response)

    # simulate propensities
    _propensity = @views [min(625.0, exp(dot(covariate[:,i], coefficient))) for i in 1:nnodes]

    # simulate Poisson expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    simulate_with_pois!(rng, observed, expected, _propensity)

    example = UndirectedMultiGraphModel(PoissonEdges(), observed, covariate)
    copyto!(example.propensity, _propensity)
    copyto!(example.coefficient, coefficient)
    copyto!(example.expected, expected)

    return example
end

function simulate_covariate_model(dist::NegBinEdges, nnodes::Int, ncovar::Int;
    dispersion::Real=1.0,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    propensity::AbstractVector=default_propensity(rng, nnodes),    
)
    # sanity check
    length(propensity) != nnodes && error("Number of nodes must match number of propensities.")
    any(<=(0), propensity) && error("Propensities must be positive.")

    # simulate p covariates for each of the m nodes; iid N(0,1)
    design_matrix = randn(rng, ncovar, nnodes)

    # center and scale
    μ = mean(design_matrix, dims=2)
    σ = std(design_matrix, dims=2)
    covariate = (design_matrix .- μ) ./ σ

    # approximately generate coefficients that produce the target propensity values
    response = log.(propensity)
    coefficient = covariate' \ (log(10, nnodes) * response)

    # simulate propensities
    _propensity = @views [min(625.0, exp(dot(covariate[:,i], coefficient))) for i in 1:nnodes]

    # set scale parameter, r = 1/dispersion
    r = 1 / dispersion

    # simulate Negative Binomial expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    simulate_with_nbin!(rng, observed, expected, _propensity, r)

    params = (scale=r, dispersion=dispersion)
    example = UndirectedMultiGraphModel(dist, observed, covariate, params)
    copyto!(example.propensity, _propensity)
    copyto!(example.coefficient, coefficient)
    copyto!(example.expected, expected)

    return example
end

export simulate_propensity_model, simulate_covariate_model

#
#   LIKELIHOOD
#

function eval_loglikelihood(model::AbstractMultiGraphModel, buffers::T=__allocate_buffers__(model)) where T
    eval_loglikelihood(model, model.covariate, buffers)
end

# no covariates
function eval_loglikelihood(model::AbstractMultiGraphModel, ::Nothing, buffers)
    __eval_loglikelihood_threaded__(model, buffers)
end

# with covariates
function eval_loglikelihood(model::AbstractMultiGraphModel, X, buffers)
    __eval_loglikelihood_and_derivs_threaded__(model, buffers)
end

# no covariates
function __eval_loglikelihood_unthreaded__(model::AbstractMultiGraphModel{distT}) where distT
    m = size(model.observed, 1)
    logl = 0.0
    for j in Base.OneTo(m), i in Base.OneTo(m)
        if i == j continue end
        Z_ij = model.observed[i,j]
        M_ij = model.expected[i,j]
        logl_ij = partial_loglikelihood(distT(), Z_ij, M_ij, model.parameters)
        if isnan(logl_ij)
            error("Detected NaN for edges between $i and $j. $Z_ij and $M_ij")
        end
        logl += logl_ij
    end
    return logl
end

function __eval_loglikelihood_threaded__(model::AbstractMultiGraphModel{distT}, buffers) where distT
    m = size(model.observed, 1)
    accumulator = buffers.accumulator[1]
    fill!(accumulator, 0)
    @batch per=core for j in Base.OneTo(m)
        local_logl = 0.0
        for i in Base.OneTo(m)
            if i == j continue end
            Z_ij = model.observed[i,j]
            M_ij = model.expected[i,j]
            logl_ij = partial_loglikelihood(distT(), Z_ij, M_ij, model.parameters)
            if isnan(logl_ij)
                error("Detected NaN for edges between $i and $j. $Z_ij and $M_ij")
            end
            local_logl += logl_ij
        end
        accumulator[Threads.threadid()] += local_logl
    end
    return sum(accumulator)
end

# with covariates
function __eval_loglikelihood_and_derivs_unthreaded__(model::AbstractMultiGraphModel{distT}, buffers) where distT
    logl = __eval_loglikelihood_unthreaded__(model)
    __eval_derivs!__(distT(), model, buffers)
    return logl
end

function __eval_loglikelihood_and_derivs_threaded__(model::AbstractMultiGraphModel{distT}, buffers) where distT
    logl = __eval_loglikelihood_threaded__(model, buffers)
    __eval_derivs!__(distT(), model, buffers)
    return logl
end

function partial_loglikelihood(::PoissonEdges, observed, expected, parameters)
    return logpdf(Poisson(expected), observed)
end

function partial_loglikelihood(::NegBinEdges{MeanScale}, observed, expected, parameters)
    # r failures, p is probability of failures
    r = parameters.scale
    p = expected / (expected + r)
    # Distributions.jl treats r as number of successes, so need to flip probability
    return logpdf(NegativeBinomial(r, 1-p), observed)
end

function partial_loglikelihood(::NegBinEdges{MeanDispersion}, observed, expected, parameters)
    a = parameters.dispersion
    p = a*expected / (a*expected + 1)
    return logpdf(NegativeBinomial(inv(a), 1-p), observed)
end

#
#   MODEL FITTING
#

init_model(model::AbstractMultiGraphModel{distT}) where distT = init_model(distT(), model)

# dispatch on distribution type, covariate type, and model type (symmetry/asymmetry)
update!(model::AbstractMultiGraphModel{distT}, buffers) where distT = update!(distT(), model.covariate, model, buffers)

init_old_state(model::AbstractMultiGraphModel) = init_old_state(model, model.covariate)

update_old_state!(state, model::AbstractMultiGraphModel) = update_old_state!(state, model, model.covariate)

backtrack_to_old_state!(model::AbstractMultiGraphModel, state) = backtrack_to_old_state!(model, state, model.covariate)

function __mle_loop__(model::AbstractMultiGraphModel, buffers, maxiter, tolerance, verbose)
    # initialize constants
    init_buffers!(model, buffers)

    init_logl = old_logl = eval_loglikelihood(model, buffers)
    iter = 0
    converged = false
    old_state = init_old_state(model)

    for _ in 1:maxiter
        iter += 1

        # Update propensities and additional parameters.
        old_state = update_old_state!(old_state, model)
        model = update!(model, buffers)

        # Evaluate model.
        logl = eval_loglikelihood(model, buffers)
        increase = logl - old_logl
        rel_tolerance = tolerance * (1 + abs(old_logl))

        # Check for ascent and convergence.
        if increase >= 0 || abs(increase) < 1e-4
            old_logl = logl
            converged = increase < rel_tolerance
            if converged
                break
            end
        else
            @warn "Ascent condition failed; exiting after $(iter) iterations." model=model loglikelihood=logl previous=old_logl
            model = backtrack_to_old_state!(model, old_state)
            break
        end
    end

    if converged && verbose
        @info "Converged after $(iter) iterations." loglikelihood=old_logl initial=init_logl
    elseif verbose
        @info "Failed to converge after $(iter) iterations." loglikelihood=old_logl initial=init_logl
    end

    result = (
        logl_init=init_logl,
        logl_final=old_logl,
        fitted=model,
        iterations=iter,
        converged=converged,
    )

    return result
end

"""
```
fit_model(dist::AbstractEdgeDistribution, observed; kwargs...)
```

Fit a propensity-based model to the `observed` data assuming a `dist` edge distribution.
"""
function fit_model(dist::AbstractEdgeDistribution, observed; directed::Bool=false, kwargs...)
    #
    if directed
        model = DirectedMultiGraphModel(dist, observed)
    else
        model = UndirectedMultiGraphModel(dist, observed)
    end
    fit_model(model; kwargs...)
end

"""
```
fit_model(dist::AbstractEdgeDistribution, observed, covariates; kwargs...)
```

Fit a `covariates`-based model to the `observed` data assuming a `dist` edge distribution.
"""
function fit_model(dist::AbstractEdgeDistribution, observed, covariates; directed::Bool=false, kwargs...)
    #
    if directed
        model = DirectedMultiGraphModel(dist, observed, covariates)
    else
        model = UndirectedMultiGraphModel(dist, observed, covariates)
    end
    fit_model(model; kwargs...)
end

"""
```
fit_model(model::MultiGraphModel{DIST}; maxiter::Real=100, tolerance::Real=1e-6) where DIST
```

Estimate the parameters of the given `model` using MLE.
"""
function fit_model(model::AbstractMultiGraphModel; maxiter::Real=100, tolerance::Real=1e-6, verbose::Bool=true)
    buffers = __allocate_buffers__(model)
    model = init_model(model)
    __mle_loop__(model, buffers, maxiter, tolerance, verbose)
end

export fit_model

#
#   ALGORITHM MAPS
#

function __mm_new_propensity!__(::PoissonEdges, p, old_p, sum_Z)
    copyto!(old_p, p)
    sum_p = sum(p)
    @batch per=core for i in eachindex(p)
        sum_p_i = sum_p - old_p[i]
        p[i] = sqrt(old_p[i] * sum_Z[i] / sum_p_i)
    end
end

function __mm_new_propensity!__(::PoissonEdges, p, old_p, q, old_q, sum_Z_row, sum_Z_col)
    copyto!(old_p, p)
    copyto!(old_q, q)
    sum_p = sum(p)
    sum_q = sum(q)
    @batch per=core for i in eachindex(p)
        sum_p_i = sum_p - old_p[i]
        sum_q_i = sum_q - old_q[i]
        p[i] = sqrt(old_p[i] * sum_Z_row[i] / sum_q_i)
        q[i] = sqrt(old_q[i] * sum_Z_col[i] / sum_p_i)
    end
end

function __mm_new_propensity!__(::NegBinEdges{MeanScale}, propensity, old_propensity, observed, expected, sum_observed, r)
    copyto!(old_propensity, propensity)
    Z = observed
    M = expected
    sum_Z = sum_observed
    @batch per=core for j in eachindex(propensity)
        denominator = 0.0
        for i in eachindex(propensity)
            # Use symmetry Z[i,j] = Z[j,i].
            if i == j continue end
            denominator += (Z[i,j]+r) / (M[i,j]+r) * old_propensity[i]
        end
        propensity[j] = sum_Z[j] / denominator
    end
end

function __mm_new_propensity!__(::NegBinEdges{MeanDispersion}, propensity, old_propensity, observed, expected, sum_observed, a)
    copyto!(old_propensity, propensity)
    Z = observed
    M = expected
    sum_Z = sum_observed
    @batch per=core for j in eachindex(propensity)
        denominator = 0.0
        for i in eachindex(propensity)
            # Use symmetry Z[i,j] = Z[j,i].
            if i == j continue end
            denominator += (a*Z[i,j]+1) / (a*M[i,j]+1) * old_propensity[i]
        end
        propensity[j] = sum_Z[j] / denominator
    end
end

function __mm_new_r_param__(model::AbstractMultiGraphModel, buffers)
    p = model.propensity
    r = model.parameters.scale
    mu = model.expected
    Z = model.observed

    A_accumulator = buffers.accumulator[1]
    B_accumulator = buffers.accumulator[2]
    fill!(A_accumulator, 0)
    fill!(B_accumulator, 0)
    digamma_r = digamma(r)

    @batch per=core for j in eachindex(p)
        local_A = 0.0
        local_B = 0.0
        for i in eachindex(p)
            if i == j continue end
            pi_ij = mu[i,j] / (mu[i,j]+r)
            local_A -= r*(digamma(Z[i,j]+r) - digamma_r)
            local_B += log1p(-pi_ij) + (mu[i,j] - Z[i,j]) / (mu[i,j] + r)
        end
        A_accumulator[Threads.threadid()] += local_A
        B_accumulator[Threads.threadid()] += local_B
    end

    return sum(A_accumulator) / sum(B_accumulator)
end

function __mm_new_a_param__(model::AbstractMultiGraphModel, buffers)
    p = model.propensity
    Z = model.observed
    a = model.parameters.dispersion
    mu = model.expected

    sum_Z = buffers.sum_Z
    B_accumulator = buffers.accumulator[1]
    fill!(B_accumulator, 0)
    digamma_inva = digamma(inv(a))

    @batch per=core for j in eachindex(p)
        local_B = 0.0
        for i in eachindex(p)
            if i == j continue end
            pi_ij = a*mu[i,j] / (a*mu[i,j]+1)
            local_B -= inv(a) * (digamma(Z[i,j] + inv(a)) - digamma_inva)
            local_B -= inv(a) * log1p(-pi_ij) + (Z[i,j] + inv(a)) * pi_ij
        end
        B_accumulator[Threads.threadid()] += local_B
    end

    return a * (-sum(sum_Z) / sum(B_accumulator))
end

function __newton_new_coefficients__(model::AbstractMultiGraphModel, buffers)
    b = model.coefficient
    v = buffers.newton_direction
    d1f = buffers.gradient
    d2f = buffers.hessian

    logl_old = __eval_loglikelihood_threaded__(model, buffers)
    H = Symmetric(d2f, :L)
    H .= H + norm(H)*eps(eltype(H))*I
    cholH = cholesky!(H)
    ldiv!(v, cholH, d1f)
    t = 1.0
    max_backtracking = 32

    for step in 0:max_backtracking
        axpy!(t, v, b)
        update_expectations!(model)
        logl_new =__eval_loglikelihood_threaded__(model, buffers)
        if logl_new > logl_old || step == max_backtracking
            break
        end
        axpy!(-t, v, b)
        t = t / 2
    end
end

end # module
