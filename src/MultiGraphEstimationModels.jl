module MultiGraphEstimationModels

using LinearAlgebra, Statistics, Distributions, SpecialFunctions
using PoissonRandom, Random, StableRNGs
using Polyester

import Base: show

abstract type AbstractEdgeDistribution end

abstract type NBParam end

struct MeanScale <: NBParam end
struct MeanDispersion <: NBParam end

struct PoissonEdges <: AbstractEdgeDistribution end

struct NegBinEdges{param<:NBParam} <: AbstractEdgeDistribution end

NegBinEdges(param::T) where T <: NBParam = NegBinEdges{T}()
NegBinEdges() = NegBinEdges(MeanScale())

function check_observed_data(observed)
    !(observed isa AbstractMatrix) && error("Observed data should enter as a matrix (suitable subtype of AbstractMatrix).")
    nrows, ncols = size(observed)
    nrows != ncols && error("Observed data is not a square matrix. Number of nodes is ambigous.")
    !(eltype(observed) <: Integer) && @warn("Observed data is not integer-valued.")
    return nothing
end

check_covariates(observed, covariates::Nothing) = nothing

function check_covariates(observed, covariates)
    nrows = size(observed, 1)
    nnodes = size(covariates, 2)
    nrows != nnodes && error("Detected number of nodes in `covariates` incompatible with `observed` matrix: $(nrows) in `observed` vs $(nnodes) in `covariates`. Note that `covariates` should enter as a (# covariates) × (# nodes matrix).")
    return nothing
end

function allocate_propensities(observed)
    p = zeros(Float64, size(observed, 1))
    return p
end

allocate_expected_matrix(observed::AbstractMatrix) = zeros(Float64, size(observed))

allocate_coefficients(covariates::AbstractMatrix) = zeros(Float64, size(covariates, 1))
allocate_coefficients(covariates::Nothing) = nothing

struct MultiGraphModel{DIST,intT,floatT,COVARIATE,COEFFICIENT,PARAMS}
    propensity::Vector{floatT}  # propensity of node i to form an edge
    coefficient::COEFFICIENT    # effect size of covariate j on node propensities
    observed::Matrix{intT}      # edge count data
    expected::Matrix{floatT}    # expected edge counts under statistical model
    covariate::COVARIATE        # nodes × covariates design matrix
    parameters::PARAMS          # additional parameters
end

function MultiGraphModel(::DIST, observed, covariate, parameters::NamedTuple) where DIST <: AbstractEdgeDistribution
    # sanity checks
    check_observed_data(observed)
    check_covariates(observed, covariate)

    # allocate data structures
    propensity = allocate_propensities(observed)
    expected = allocate_expected_matrix(observed)
    coefficient = allocate_coefficients(covariate)

    # determine type parameters
    COVARIATE = typeof(covariate)
    COEFFICIENT = typeof(coefficient)
    PARAMS = typeof(parameters)
    intT = eltype(observed)
    floatT = eltype(propensity)

    model = MultiGraphModel{DIST,intT,floatT,COVARIATE,COEFFICIENT,PARAMS}(propensity, coefficient, observed, expected, covariate, parameters)
    update_expectations!(model)
    return model
end

function MultiGraphModel(dist::AbstractEdgeDistribution, observed)
    MultiGraphModel(dist, observed, nothing)
end

function MultiGraphModel(dist::AbstractEdgeDistribution, observed, params::NamedTuple)
    MultiGraphModel(dist, observed, nothing, params)
end

function MultiGraphModel(dist::PoissonEdges, observed, covariates)
    params = (scale=Inf, dispersion=0.0)
    MultiGraphModel(dist, observed, covariates, params)
end

function MultiGraphModel(dist::NegBinEdges, observed, covariates)
    r = Inf
    params = (scale=r, dispersion=1/r)
    MultiGraphModel(dist, observed, covariates, params)
end

function Base.show(io::IO, model::MultiGraphModel{DIST,intT,floatT}) where {DIST,intT,floatT}
    nnodes = length(model.propensity)
    ncovar = model.coefficient isa Nothing ? 0 : length(model.coefficient)

    print(io, "MultiGraphModel{$(intT),$(floatT)}:")
    print(io, "\n  - distribution: $(DIST)")
    print(io, "\n  - nodes: $(nnodes)")
    print(io, "\n  - covariates: $(ncovar)")

    return nothing
end

function update_expectations!(model::MultiGraphModel)
    update_expectations!(model, model.covariate)
end

function update_expectations!(model::MultiGraphModel, ::Any)
    p = model.propensity
    Z = model.covariate
    b = model.coefficient
    mul!(p, transpose(Z), b)
    @. p = min(625.0, exp(p))
    update_expectations!(model, nothing)
end

function update_expectations!(model::MultiGraphModel, ::Nothing)
    p = model.propensity
    mu = model.expected
    T = eltype(mu)
    @batch per=core for j in eachindex(p)
        for i in eachindex(p)
            if i == j continue end
            mu[i,j] = p[i] * p[j]
        end
    end
end

function remake_model!(model::MultiGraphModel{DIST}, new_params) where DIST
    intT, floatT = eltype(model.observed), eltype(model.expected)
    COV, COF, PAR = typeof(model.covariate), typeof(model.coefficient), typeof(new_params)
    return MultiGraphModel{DIST,intT,floatT,COV,COF,PAR}(
        model.propensity,
        model.coefficient,
        model.observed,
        model.expected,
        model.covariate,
        new_params,
    )
end

export PoissonEdges, NBParam, MeanScale, MeanDispersion, NegBinEdges, MultiGraphModel

#
#   SIMULATION
#

function simulate_propensity_model(::PoissonEdges, nnodes::Int; seed::Integer=1903)
    # initialize RNG
    rng = StableRNG(seed)

    # draw propensities uniformly from (0, 10)
    propensity = 10*rand(rng, nnodes)

    # simulate Poisson expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    for j in 1:nnodes, i in j+1:nnodes
        mu_ij = propensity[i] * propensity[j]
        observed[i,j] = observed[j,i] = pois_rand(rng, mu_ij)
        expected[i,j] = expected[j,i] = mu_ij
    end

    example = MultiGraphModel(PoissonEdges(), observed)
    copyto!(example.propensity, propensity)
    copyto!(example.expected, expected)

    return example
end

function simulate_propensity_model(dist::NegBinEdges, nnodes::Int; dispersion::Real=1.0, seed::Integer=1903)
    # initialize RNG
    rng = StableRNG(seed)

    # draw propensities uniformly from (0, 10)
    propensity = 10*rand(rng, nnodes)

    # set scale parameter, r = 1/dispersion
    r = 1 / dispersion

    # simulate Negative Binomial expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    for j in 1:nnodes, i in j+1:nnodes
        mu_ij = propensity[i] * propensity[j]
        pi_ij = mu_ij / (mu_ij + r)
        D = NegativeBinomial(r, 1-pi_ij)
        observed[i,j] = observed[j,i] = rand(rng, D)
        expected[i,j] = expected[j,i] = mu_ij
    end

    params = (scale=r, dispersion=dispersion)
    example = MultiGraphModel(dist, observed, nothing, params)
    copyto!(example.propensity, propensity)
    copyto!(example.expected, expected)

    return example
end

function simulate_covariate_model(::PoissonEdges, nnodes::Int, ncovar::Int; seed::Integer=1903)
    # initialize RNG
    rng = StableRNG(seed)

    # simulate p covariates for each of the m nodes; iid N(0,1)
    design_matrix = randn(rng, ncovar, nnodes)

    # center and scale
    μ = mean(design_matrix, dims=2)
    σ = std(design_matrix, dims=2)
    covariate = (design_matrix .- μ) ./ σ

    # approximately generate coefficients that produce propensities in [0,10]
    response = -randexp(rng, nnodes) .+ log(10)
    coefficient = covariate' \ (log(10, nnodes) * response)

    # simulate propensities
    propensity = @views [min(625.0, exp(dot(covariate[:,i], coefficient))) for i in 1:nnodes]

    # simulate Poisson expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    for j in 1:nnodes, i in j+1:nnodes
        mu_ij = propensity[i] * propensity[j]
        observed[i,j] = observed[j,i] = pois_rand(rng, mu_ij)
        expected[i,j] = expected[j,i] = mu_ij
    end

    example = MultiGraphModel(PoissonEdges(), observed, covariate)
    copyto!(example.propensity, propensity)
    copyto!(example.coefficient, coefficient)
    copyto!(example.expected, expected)

    return example
end

function simulate_covariate_model(dist::NegBinEdges, nnodes::Int, ncovar::Int; dispersion::Real=1.0, seed::Integer=1903)
    # initialize RNG
    rng = StableRNG(seed)

    # simulate p covariates for each of the m nodes; iid N(0,1)
    design_matrix = randn(rng, ncovar, nnodes)

    # center and scale
    μ = mean(design_matrix, dims=2)
    σ = std(design_matrix, dims=2)
    covariate = (design_matrix .- μ) ./ σ

    # approximately generate coefficients that produce propensities in [0,10]
    response = -log(10, nnodes) .* log.(randn(rng, nnodes) .^ 2)
    coefficient = covariate' \ response

    # simulate propensities
    propensity = @views [min(625.0, exp(dot(covariate[:,i], coefficient))) for i in 1:nnodes]

    # set scale parameter, r = 1/dispersion
    r = 1 / dispersion

    # simulate Negative Binomial expectations and data
    expected = zeros(Float64, nnodes, nnodes)
    observed = zeros(Int, nnodes, nnodes)
    for j in 1:nnodes, i in j+1:nnodes
        mu_ij = propensity[i] * propensity[j]
        pi_ij = mu_ij / (mu_ij + r)
        D = NegativeBinomial(r, 1-pi_ij)
        observed[i,j] = observed[j,i] = rand(rng, D)
        expected[i,j] = expected[j,i] = mu_ij
    end

    params = (scale=r, dispersion=dispersion)
    example = MultiGraphModel(dist, observed, covariate, params)
    copyto!(example.propensity, propensity)
    copyto!(example.coefficient, coefficient)
    copyto!(example.expected, expected)

    return example
end

export simulate_propensity_model, simulate_covariate_model

#
#   LIKELIHOOD
#

# no buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}) where DIST
    eval_loglikelihood(model, model.covariate, nothing)
end

# with buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, buffers) where DIST
    eval_loglikelihood(model, model.covariate, buffers)
end

# no covariates, no buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, ::Nothing, ::Nothing) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, nothing)
    __eval_loglikelihood_threaded__(model, buffers)
end

# with covariates, allocate buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, Z, ::Nothing) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, Z)
    __eval_loglikelihood_and_derivs_threaded__(model, buffers)
end

# no covariates, with buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, ::Nothing, buffers) where DIST
    __eval_loglikelihood_threaded__(model, buffers)
end

# with covariates, with buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, Z, buffers) where DIST
    __eval_loglikelihood_and_derivs_threaded__(model, buffers)
end

# no covariates
function __eval_loglikelihood_unthreaded__(model::MultiGraphModel{DIST}) where DIST
    p = model.propensity
    logl = 0.0
    for j in eachindex(p)
        for i in eachindex(p)
            if i == j continue end
            x_ij = model.observed[i,j]
            mu_ij = model.expected[i,j]
            logl_ij = partial_loglikelihood(DIST(), x_ij, mu_ij, model.parameters)
            if isnan(logl_ij)
                error("Detected NaN for edges between $i and $j. $x_ij and $mu_ij")
            end
            logl += logl_ij
        end
    end
    return logl
end

function __eval_loglikelihood_threaded__(model::MultiGraphModel{DIST}, buffers) where DIST
    p = model.propensity
    accumulator = buffers.accumulator[1]
    fill!(accumulator, 0)
    @batch per=core for j in eachindex(p)
        local_logl = 0.0
        for i in eachindex(p)
            if i == j continue end
            x_ij = model.observed[i,j]
            mu_ij = model.expected[i,j]
            logl_ij = partial_loglikelihood(DIST(), x_ij, mu_ij, model.parameters)
            if isnan(logl_ij)
                error("Detected NaN for edges between $i and $j. $x_ij and $mu_ij")
            end
            local_logl += logl_ij
        end
        accumulator[Threads.threadid()] += local_logl
    end
    return sum(accumulator)
end

# with covariates
function __eval_loglikelihood_and_derivs_unthreaded__(model::MultiGraphModel{DIST}, buffers) where DIST
    logl = __eval_loglikelihood_unthreaded__(model)
    __eval_derivs!__(DIST(), model, buffers)
    return logl
end

function __eval_loglikelihood_and_derivs_threaded__(model::MultiGraphModel{DIST}, buffers) where DIST
    logl = __eval_loglikelihood_threaded__(model, buffers)
    __eval_derivs!__(DIST(), model, buffers)
    return logl
end

function __eval_derivs!__(::PoissonEdges, model, buffers)
    # Assumes Poisson
    d1f = buffers.gradient
    d2f = buffers.hessian
    tmp1 = buffers.m_by_m
    tmp2 = buffers.c_by_m
    w = buffers.diag
    X = model.covariate
    M = model.expected
    Z = model.observed

    # gradient
    @. tmp1 = Z - M
    mul!(tmp2, X, tmp1)
    sum!(d1f, tmp2)
    @. d1f = 2*d1f

    # Hessian
    sum!(w, M)
    @. tmp1 = $Diagonal(w) + M
    mul!(transpose(tmp2), tmp1, transpose(X))
    mul!(d2f, X, transpose(tmp2))
    @. d2f = 2*d2f

    return nothing
end

function __eval_derivs!__(::NegBinEdges{MeanScale}, model, buffers)
    # Assumes mean-scale
    d1f = buffers.gradient
    d2f = buffers.hessian
    tmp1 = buffers.m_by_m
    tmp2 = buffers.c_by_m
    w = buffers.diag
    X = model.covariate
    M = model.expected
    Z = model.observed
    r = model.parameters.scale

    # gradient
    @. tmp1 = Z - (Z+r)/(M+r)*M # element-wise operations
    mul!(tmp2, X, tmp1)
    sum!(d1f, tmp2)
    @. d1f = 2*d1f

    # Hessian
    @. tmp1 = (1 - M/(M+r))*M # element-wise operations
    sum!(w, tmp1)
    @. tmp1 = $Diagonal(w) + tmp1
    mul!(transpose(tmp2), tmp1, transpose(X))
    mul!(d2f, X, transpose(tmp2))
    @. d2f = 2*d2f

    return nothing
end

function __eval_derivs!__(::NegBinEdges{MeanDispersion}, model, buffers)
    # Assumes mean-dispersion
    d1f = buffers.gradient
    d2f = buffers.hessian
    tmp1 = buffers.m_by_m
    tmp2 = buffers.c_by_m
    w = buffers.diag
    X = model.covariate
    M = model.expected
    Z = model.observed
    a = model.parameters.dispersion

    # gradient
    @. tmp1 = Z - (a*Z+1)/(a*M+1)*M # element-wise operations
    mul!(tmp2, X, tmp1)
    sum!(d1f, tmp2)
    @. d1f = 2*d1f

    # Hessian
    @. tmp1 = (1 - a*M/(a*M+1))*M # element-wise operations
    sum!(w, tmp1)
    @. tmp1 = $Diagonal(w) + tmp1
    mul!(transpose(tmp2), tmp1, transpose(X))
    mul!(d2f, X, transpose(tmp2))
    @. d2f = 2*d2f

    return nothing
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

function partial_gradient!(::PoissonEdges, d1f, tmp, observed, expected, zi, zj)
    α = observed - expected
    @. tmp = zi + zj
    
    # update ∇logl = ∇logl + α(xᵢ+xⱼ)
    axpy!(α, tmp, d1f)

    return d1f
end

function partial_hessian!(::PoissonEdges, d2f, tmp, observed, expected, zi, zj)
    α = -expected
    @. tmp = zi + zj
    
    # update ∇²logl = ∇²logl + α (xᵢ+xⱼ) (xᵢ+xⱼ)ᵀ
    # only on the upper triangular half
    BLAS.syr!('L', α, tmp, d2f)

    return d2f
end

#
#   MODEL FITTING
#

# Case: PoissonEdges, no covariates
function __allocate_buffers__(::PoissonEdges, p, ::Nothing)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:1],
    )
    return buffers
end

# Case: NegBinEdges, no covariates
function __allocate_buffers__(::NegBinEdges, p, ::Nothing)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:2],
    )
    return buffers
end

# Case: PoissonEdges, with covariates
function __allocate_buffers__(::PoissonEdges, p, Z::AbstractMatrix)
    nnodes = length(p)
    ncovars = size(Z, 1)
    nt = Threads.nthreads()
    buffers = (;
        sum_x=zeros(Int, nnodes),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:1],
        newton_direction=zeros(ncovars),
        gradient=zeros(ncovars),
        hessian=zeros(ncovars, ncovars),
        m_by_m=zeros(nnodes, nnodes),
        c_by_m=zeros(ncovars, nnodes),
        diag=zeros(nnodes),
    )
    return buffers
end

# Case: NegBinEdges, with covariates
function __allocate_buffers__(::NegBinEdges, p, Z::AbstractMatrix)
    nnodes = length(p)
    ncovars = size(Z, 1)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:2],
        newton_direction=zeros(ncovars),
        gradient=zeros(ncovars),
        hessian=zeros(ncovars, ncovars),
        m_by_m=zeros(nnodes, nnodes),
        c_by_m=zeros(ncovars, nnodes),
        diag=zeros(nnodes),
    )
    return buffers
end

function __mle_loop__(model, buffers, maxiter, tolerance, verbose)
    # initialize constants
    sum!(buffers.sum_x, model.observed)

    init_logl = old_logl = eval_loglikelihood(model, buffers)
    iter = 0
    converged = false    

    for _ in 1:maxiter
        iter += 1

        # Update propensities and additional parameters.
        model = update!(model, buffers)

        # Evaluate model.
        logl = eval_loglikelihood(model, buffers)
        increase = logl - old_logl
        rel_tolerance = tolerance * (1 + abs(old_logl))

        # Check for ascent and convergence.
        if increase >= 0
            old_logl = logl
            converged = increase < rel_tolerance
            if converged
                break
            end
        else
            @warn "Ascent condition failed; exiting after $(iter) iterations." model=model loglikelihood=logl previous=old_logl
            old_logl = logl
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

function init_model(::PoissonEdges, model)
    m = length(model.propensity)
    sum_x = sum(model.observed)
    model.propensity .= sqrt( sum_x / (m * (m-1)) )
    if !(model.covariate isa Nothing)
        A = copy(model.covariate)       # LHS
        b = log.(model.propensity)      # RHS
        model.coefficient .= A' \ b
    end
    update_expectations!(model)
    return model
end

function init_model(::NegBinEdges, model)
    # initialize with rough estimates under Poisson model
    init = MultiGraphModel(PoissonEdges(), model.observed, model.covariate)
    result = fit_model(init; maxiter=5, verbose=false)
    if !(model.covariate isa Nothing)
        copyto!(model.coefficient, result.fitted.coefficient)
    else
        copyto!(model.propensity, result.fitted.propensity)
    end
    update_expectations!(model)

    # use MoM estimator for r
    nzdata = eltype(model.observed)[]
    for j in axes(model.observed, 2), i in 1:j-1
        push!(nzdata, model.observed[i,j])
    end
    xbar = mean(nzdata)
    s2 = var(nzdata, mean=xbar)
    r_init = xbar^2 / (s2 - xbar)
    if r_init < 0
        @warn "Initialization of scale parameter r using method of moments failed. Defaulting to r=1." sample_mean=xbar sample_variance=s2
        r_init = one(r_init)
    end

    # calibrate by searching over logarithmic grid
    best_r, best_logl = r_init, -Inf
    for x in range(-3, 3, step=0.5)
        r = r_init * 10.0 ^ x
        model = remake_model!(model, (scale=r, dispersion=inv(r)))
        update_expectations!(model)
        logl = eval_loglikelihood(model, nothing, nothing)
        if logl > best_logl
            best_r, best_logl = r, logl
        end
    end
    model = remake_model!(model, (scale=best_r, dispersion=inv(best_r)))
    update_expectations!(model)
    return model
end

"""
```
fit_model(dist::AbstractEdgeDistribution, observed; kwargs...)
```

Fit a propensity-based model to the `observed` data assuming a `dist` edge distribution.
"""
function fit_model(dist::AbstractEdgeDistribution, observed; kwargs...)
    #
    model = MultiGraphModel(dist, observed)
    fit_model(model; kwargs...)
end

"""
```
fit_model(dist::AbstractEdgeDistribution, observed, covariates; kwargs...)
```

Fit a `covariates`-based model to the `observed` data assuming a `dist` edge distribution.
"""
function fit_model(dist::AbstractEdgeDistribution, observed, covariates; kwargs...)
    #
    model = MultiGraphModel(dist, observed, covariates)
    fit_model(model; kwargs...)
end

"""
```
fit_model(model::MultiGraphModel{DIST}; maxiter::Real=100, tolerance::Real=1e-6) where DIST
```

Estimate the parameters of the given `model` using MLE.
"""
function fit_model(model::MultiGraphModel{DIST}; maxiter::Real=100, tolerance::Real=1e-6, verbose::Bool=true) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, model.covariate)
    model = init_model(DIST(), model)
    __mle_loop__(model, buffers, maxiter, tolerance, verbose)
end

export fit_model

#
#   ALGORITHM MAPS
#

function __mm_new_r_param__(model, buffers)
    p = model.propensity
    r = model.parameters.scale
    mu = model.expected
    x = model.observed

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
            local_A -= r*(digamma(x[i,j]+r) - digamma_r)
            local_B += log1p(-pi_ij) + (mu[i,j] - x[i,j]) / (mu[i,j] + r)
        end
        A_accumulator[Threads.threadid()] += local_A
        B_accumulator[Threads.threadid()] += local_B
    end

    return sum(A_accumulator) / sum(B_accumulator)
end

function __mm_new_a_param__(model, buffers)
    p = model.propensity
    x = model.observed
    a = model.parameters.dispersion
    mu = model.expected

    sum_x = buffers.sum_x
    B_accumulator = buffers.accumulator[1]
    fill!(B_accumulator, 0)
    digamma_inva = digamma(inv(a))

    @batch per=core for j in eachindex(p)
        local_B = 0.0
        for i in eachindex(p)
            if i == j continue end
            pi_ij = a*mu[i,j] / (a*mu[i,j]+1)
            local_B -= inv(a) * (digamma(x[i,j] + inv(a)) - digamma_inva)
            local_B -= inv(a) * log1p(-pi_ij) + (x[i,j] + inv(a)) * pi_ij
        end
        B_accumulator[Threads.threadid()] += local_B
    end

    return a * (-sum(sum_x) / sum(B_accumulator))
end

function __newton_new_coefficients__(model, buffers)
    b = model.coefficient
    v = buffers.newton_direction
    d1f = buffers.gradient
    d2f = buffers.hessian

    logl_old = __eval_loglikelihood_threaded__(model, buffers)
    cholH = cholesky!(Symmetric(d2f, :L))
    ldiv!(v, cholH, d1f)
    t = 1.0
    max_backtracking = 8
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

update!(model::MultiGraphModel{DIST}, buffers) where DIST = update!(DIST(), model.covariate, model, buffers)

# Case: PoissonEdges, no covariates
function update!(::PoissonEdges, ::Nothing, model, buffers)
    p = model.propensity

    sum_x = buffers.sum_x
    sum_p = sum(p)
    old_p = copyto!(buffers.old_p, p)

    @batch per=core for i in eachindex(p)
        sum_p_i = sum_p - old_p[i]
        p[i] = sqrt(old_p[i] * sum_x[i] / sum_p_i)
    end

    update_expectations!(model)

    return model
end

# Case: NegBinEdges, mean-scale, no covariates
function update!(::NegBinEdges{MeanScale}, ::Nothing, model, buffers)
    p = model.propensity
    x = model.observed
    r = model.parameters.scale
    mu = model.expected

    sum_x = buffers.sum_x
    old_p = copyto!(buffers.old_p, p)

    # Update propensities.
    @batch per=core for j in eachindex(p)
        denominator = 0.0
        for i in eachindex(p)
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            denominator += (x[i,j]+r) / (mu[i,j]+r) * old_p[i]
        end
        p[j] = sum_x[j] / denominator
    end
    update_expectations!(model)

    # Update scale parameter, r.
    new_r = __mm_new_r_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=new_r, dispersion=inv(new_r))
    return remake_model!(model, new_parameters)
end

# Case: NegBinEdges, mean-dispersion, no covariates
function update!(::NegBinEdges{MeanDispersion}, ::Nothing, model, buffers)
    p = model.propensity
    x = model.observed
    a = model.parameters.dispersion
    mu = model.expected

    sum_x = buffers.sum_x
    old_p = copyto!(buffers.old_p, p)

    # Update propensities.
    @batch per=core for j in eachindex(p)
        denominator = 0.0
        for i in eachindex(p)
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            denominator += (a*x[i,j]+1) / (a*mu[i,j]+1) * old_p[i]
        end
        p[j] = sum_x[j] / denominator
    end
    update_expectations!(model)

    # Update dispersion parameter, a.
    new_a = __mm_new_a_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=inv(new_a), dispersion=new_a)
    return remake_model!(model, new_parameters)
end

# Case: PoissonEdges, with covariates
function update!(::PoissonEdges, ::AbstractMatrix, model, buffers)
    # Update coefficients with Newton's method
    __newton_new_coefficients__(model, buffers)
    update_expectations!(model)

    return model
end

# Case: NegBinEdges, mean-scale, with covariates
function update!(::NegBinEdges{MeanScale}, ::AbstractMatrix, model, buffers)
    # Update coefficients with Newton's method
    __newton_new_coefficients__(model, buffers)
    update_expectations!(model)

    # Update scale parameter, r.
    new_r = __mm_new_r_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=new_r, dispersion=inv(new_r))
    return remake_model!(model, new_parameters)
end

# Case: NegBinEdges, mean-dispersion, with covariates
function update!(::NegBinEdges{MeanDispersion}, ::AbstractMatrix, model, buffers)
    # Update coefficients with Newton's method
    __newton_new_coefficients__(model, buffers)
    update_expectations!(model)

    # Update dispersion parameter, a.
    new_a = __mm_new_a_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=inv(new_a), dispersion=new_a)
    return remake_model!(model, new_parameters)
end

end # module
