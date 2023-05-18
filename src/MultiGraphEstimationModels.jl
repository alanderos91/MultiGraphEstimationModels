module MultiGraphEstimationModels

using LinearAlgebra, Statistics, Distributions, SpecialFunctions
using PoissonRandom, Random, StableRNGs

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
    m = length(p)
    sum_x = sum(observed)
    @. p = sqrt( sum_x / (m * (m-1)) )
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
    nzdata = eltype(observed)[]
    for j in axes(observed, 2), i in axes(observed, 1)
        if i == j continue end
        push!(nzdata, observed[i,j])
    end
    xbar = mean(nzdata)
    s2 = var(nzdata, mean=xbar)
    r = xbar^2 / (s2 - xbar)
    if r < 0
        @warn "Initialization of scale parameter r using method of moments failed" sample_mean=xbar sample_variance=s2 
    end
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
    Threads.@threads for j in eachindex(p)
        p_j = p[j]
        v = view(mu, :, j)
        for i in eachindex(p)
            if i == j continue end
            v[i] = p[i] * p_j
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

    # simulate Poisson expectations and data
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

    # draw effect sizes uniformly from [-3, 3] 
    coefficient = zeros(ncovar)
    coefficient[1] = 3*(2*rand(rng)-1)
    coefficient[2:ncovar] = 1e-1*randn(rng, ncovar-1)

    # simulate propensities
    propensity = @views [exp(dot(covariate[:,i], coefficient)) for i in 1:nnodes]

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

# no covariates, allocate buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, ::Nothing, ::Nothing) where DIST
    __eval_loglikelihood_threaded__(model, zeros(Threads.nthreads()))
end

# with covariates, allocate buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, Z, ::Nothing) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, Z)
    __eval_loglikelihood_and_derivs_threaded__(model, buffers)
end

# no covariates, with buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, ::Nothing, buffers) where DIST
    __eval_loglikelihood_threaded__(model, buffers.threads_buffers[1])
end

# with covariates, with buffers
function eval_loglikelihood(model::MultiGraphModel{DIST}, Z, buffers) where DIST
    __eval_loglikelihood_and_derivs_threaded__(model, buffers)
end

# no covariates
function __eval_loglikelihood_threaded__(model::MultiGraphModel{DIST}, buffer) where DIST
    fill!(buffer, 0)
    p = model.propensity
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        for i in eachindex(p)
            if i == j continue end
            x_ij = model.observed[i,j]
            mu_ij = model.expected[i,j]
            logl_ij = partial_loglikelihood(DIST(), x_ij, mu_ij, model.parameters)
            if isnan(logl_ij)
                error("Detected NaN for edges between $i and $j. $x_ij and $mu_ij")
            end
            buffer[t] += logl_ij
        end
    end
    return sum(buffer)
end

# with covariates
function __eval_loglikelihood_and_derivs_threaded__(model::MultiGraphModel{DIST}, buffers) where DIST
    logl = fill!(buffers.threads_buffers[1], 0)
    d1f = buffers.threads_gradient
    d2f = buffers.threads_hessian
    tmp = buffers.threads_tmp
    foreach(Base.Fix2(fill!, 0), d1f)
    foreach(Base.Fix2(fill!, 0), d2f)

    p = model.propensity
    covariates = model.covariate
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        zj = view(covariates, :, j)
        for i in eachindex(p)
            if i == j continue end
            x_ij = model.observed[i,j]
            mu_ij = model.expected[i,j]
            zi = view(covariates, :, i)
            logl_ij = partial_loglikelihood(DIST(), x_ij, mu_ij, model.parameters)
            if isnan(logl_ij)
                error("Detected NaN for edges between $i and $j. $x_ij and $mu_ij")
            end
            logl[t] += logl_ij
            partial_gradient!(DIST(), d1f[t], tmp[t], x_ij, mu_ij, zi, zj)
            partial_hessian!(DIST(), d2f[t], tmp[t], x_ij, mu_ij, zi, zj)
        end
    end

    T = eltype(p)
    fill!(buffers.gradient, 0)
    fill!(buffers.hessian, 0)
    for k in eachindex(d1f)
        axpy!(one(T), d1f[k], buffers.gradient)
        axpy!(-one(T), d2f[k], buffers.hessian)
    end
    return sum(logl)
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
        threads_buffers=[zeros(Threads.nthreads()) for _ in 1:1]
    )
    return buffers
end

# Case: NegBinEdges, no covariates
function __allocate_buffers__(::NegBinEdges, p, ::Nothing)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        threads_buffers=[zeros(Threads.nthreads()) for _ in 1:2],
    )
    return buffers
end

# Case: PoissonEdges, with covariates
function __allocate_buffers__(::PoissonEdges, p, Z::AbstractMatrix)
    ncovars = size(Z, 1)
    nt = Threads.nthreads()
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        newton_direction=zeros(ncovars),
        gradient=zeros(ncovars),
        hessian=zeros(ncovars, ncovars),
        threads_buffers=[zeros(Threads.nthreads()) for _ in 1:1],
        threads_gradient=[zeros(ncovars) for _ in 1:nt],
        threads_hessian=[zeros(ncovars, ncovars) for _ in 1:nt],
        threads_tmp=[zeros(ncovars) for _ in 1:nt],
    )
    return buffers
end

function __mle_loop__(model, buffers, maxiter, tolerance)
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
        if increase > 0
            old_logl = logl
            converged = increase < rel_tolerance
            if converged
                break
            end
        else
            @warn "Ascent condition failed; exiting after $(iter) iterations." loglikelihood=logl previous=old_logl
            old_logl = logl
            break
        end
    end

    if converged
        @info "Converged after $(iter) iterations." loglikelihood=old_logl initial=init_logl
    else
        @info "Failed to converge after $(iter) iterations." loglikelihood=old_logl initial=init_logl
    end

    return model
end

function init_model(::PoissonEdges, model)
    if !(model.covariate isa Nothing)
        m = length(model.propensity)
        sum_x = sum(model.observed)
        model.propensity .= sqrt( sum_x / (m * (m-1)) )

        A = copy(model.covariate)       # LHS
        b = A * log.(model.propensity)  # RHS
        qrA = qr!(A')                   # in-place QR decomposition of A
        R = LowerTriangular(qrA.R)      # R factor, wrapped as UpperTriangular for dispatch
        y = R' \ b                      # Solve R'y = b
        ldiv!(model.coefficient, R, y)  # Solve R*x = y
    end
    update_expectations!(model)
    return model
end

function init_model(::NegBinEdges, model)
    init = MultiGraphModel(PoissonEdges(), model.observed, model.covariate)
    copyto!(init.propensity, model.propensity)
    update_expectations!(init)
    init = fit_model(init)
    copyto!(model.propensity, init.propensity)
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
function fit_model(model::MultiGraphModel{DIST}; maxiter::Real=100, tolerance::Real=1e-6) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, model.covariate)
    model = init_model(DIST(), model)
    __mle_loop__(model, buffers, maxiter, tolerance)
end

export fit_model

#
#   ALGORITHM MAPS
#

update!(model::MultiGraphModel{DIST}, buffers) where DIST = update!(DIST(), model.covariate, model, buffers)

# Case: PoissonEdges, no covariates
function update!(::PoissonEdges, ::Nothing, model, buffers)
    p = model.propensity

    sum_x = buffers.sum_x
    sum_p = sum(p)
    old_p = copyto!(buffers.old_p, p)

    Threads.@threads for i in eachindex(p)
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
    storage = buffers.threads_buffers[1]

    # Update propensities.
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        storage[t] = 0
        for i in eachindex(p)
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            storage[t] += (x[i,j]+r) / (mu[i,j]+r) * old_p[i]
        end
        p[j] = sum_x[j] / storage[t]
    end
    update_expectations!(model)

    # Update scale parameter, r.
    A = fill!(buffers.threads_buffers[1], 0)
    B = fill!(buffers.threads_buffers[2], 0)
    digamma_r = digamma(r)
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        u = view(x, :, j)
        v = view(mu, :, j)
        for i in eachindex(p)
            if i == j continue end
            pi_ij = v[i] / (v[i]+r)
            A[t] -= r*(digamma(u[i]+r) - digamma_r)
            B[t] += log1p(-pi_ij) + (v[i] - u[i]) / (v[i] + r)
        end
    end
    new_r = sum(A) / sum(B)

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
    storage = buffers.threads_buffers[1]

    # Update propensities.
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        storage[t] = 0
        for i in eachindex(p)
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            storage[t] += (a*x[i,j]+1) / (a*mu[i,j]+1) * old_p[i]
        end
        p[j] = sum_x[j] / storage[t]
    end
    update_expectations!(model)

    # Update dispersion parameter, a.
    A = fill!(buffers.threads_buffers[1], 0)
    B = fill!(buffers.threads_buffers[2], 0)
    digamma_inva = digamma(inv(a))
    Threads.@threads for j in eachindex(p)
        t = Threads.threadid()
        u = view(x, :, j)
        v = view(mu, :, j)
        for i in eachindex(p)
            if i == j continue end
            pi_ij = a*v[i] / (a*v[i]+1)
            A[t] -= u[i]
            B[t] -= inv(a)*(digamma(u[i]+inv(a)) - digamma_inva)
            B[t] -= inv(a) * log1p(-pi_ij) + (u[i] + inv(a))*pi_ij
        end
    end
    new_a = a * (sum(A) / sum(B))

    update_expectations!(model)
    new_parameters = (scale=inv(new_a), dispersion=new_a)
    return remake_model!(model, new_parameters)
end

# Case: PoissonEdges, with covariates
function update!(::PoissonEdges, ::AbstractMatrix, model, buffers)
    b = model.coefficient
    v = buffers.newton_direction
    d1f = buffers.gradient
    d2f = buffers.hessian
    T = eltype(v)

    # Update propensities with Newton's method
    logl_old = __eval_loglikelihood_threaded__(model, buffers.threads_buffers[1])
    cholH = cholesky!(Symmetric(d2f, :L))
    ldiv!(v, cholH, d1f)
    t = 1.0
    for step in 0:8
        axpy!(t, v, b)
        update_expectations!(model)
        logl_new =__eval_loglikelihood_threaded__(model, buffers.threads_buffers[1])
        if logl_new > logl_old || step == 4
            break
        end
        axpy!(-t, v, b)
        t = t / 2
    end

    return model
end

end # module
