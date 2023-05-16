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
    p = model.propensity
    m = length(p)
    mu = model.expected
    for j in 1:m, i in 1:m
        if i == j continue end
        mu[i,j] = p[i] * p[j]
    end
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

function loglikelihood(model::MultiGraphModel{DIST}) where DIST
    logl = 0.0
    m = length(model.propensity)
    for j in 1:m, i in 1:m
        if i == j continue end
        x_ij = model.observed[i,j]
        mu_ij = model.expected[i,j]
        logl_ij = partial_loglikelihood(DIST(), x_ij, mu_ij, model.parameters)
        if isnan(logl_ij)
            error("Detected NaN for edges between $i and $j. $x_ij and $mu_ij")
        end
        logl += logl_ij
    end
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

# Case: PoissonEdges, no covariates
function __allocate_buffers__(::PoissonEdges, p, ::Nothing)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p)
    )
    return buffers
end

# Case: NegBinEdges, no covariates
function __allocate_buffers__(::NegBinEdges, p, ::Nothing)
    buffers = (;
        sum_x=zeros(Int, length(p)),
        old_p=similar(p),
        threads_buffer=zeros(Threads.nthreads())
    )
    return buffers
end

function __mle_loop__(model, buffers, maxiter, tolerance)
    #
    init_logl = old_logl = loglikelihood(model)
    iter = 0
    converged = false

    for _ in 1:maxiter
        iter += 1

        # Update propensities and additional parameters.
        model = update!(model, buffers)

        # Evaluate model.
        logl = loglikelihood(model)
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

function fit_model(dist::AbstractEdgeDistribution, observed; kwargs...)
    #
    model = MultiGraphModel(dist, observed)
    fit_model(model; kwargs...)
end

function fit_model(dist::AbstractEdgeDistribution, observed, covariates; kwargs...)
    #
    model = MultiGraphModel(dist, observed, covariates)
    fit_model(model; kwargs...)
end

function fit_model(model::MultiGraphModel{DIST}; maxiter::Real=100, tolerance::Real=1e-6) where DIST
    buffers = __allocate_buffers__(DIST(), model.propensity, model.covariate)
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
    m = length(p)
    x = model.observed

    # Assume x[i,i] = 0. Is this done in parallel? Ideally we should compute outside function...
    sum_x = sum!(buffers.sum_x, x)
    sum_p = sum(p)
    old_p = copyto!(buffers.old_p, p)

    Threads.@threads for i in 1:m
        sum_p_i = sum_p - old_p[i]
        p[i] = sqrt(old_p[i] * sum_x[i] / sum_p_i)
    end

    update_expectations!(model)

    return model
end

# Case: NegBinEdges, mean-scale, no covariates
function update!(::NegBinEdges{MeanScale}, ::Nothing, model, buffers)
    p = model.propensity
    m = length(p)
    x = model.observed
    r = model.parameters.scale
    mu = model.expected

    sum_x = sum!(buffers.sum_x, x)
    old_p = copyto!(buffers.old_p, p)
    storage = buffers.threads_buffer

    # Update propensities.
    Threads.@threads for j in 1:m
        t = Threads.threadid()
        storage[t] = 0
        for i in 1:m
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            storage[t] += (x[i,j]+r) / (mu[i,j]+r) * old_p[i]
        end
        p[j] = sum_x[j] / storage[t]
    end
    update_expectations!(model)

    # Update scale parameter, r.
    A, B = 0.0, 0.0
    for j in 1:m, i in 1:m
        if i == j continue end
        for k in 0:(x[i,j]-1)
            A -= r / (r+k)
        end
        pi_ij = mu[i,j] / (mu[i,j]+r)
        B += log1p(-pi_ij) + (mu[i,j] - x[i,j]) / (mu[i,j] + r)
    end
    r = A / B

    update_expectations!(model)
    new_parameters = (scale=r, dispersion=inv(r))
    intT, floatT, COV, COF, PAR = eltype(x), eltype(mu), typeof(model.covariate), typeof(model.coefficient), typeof(new_parameters)
    return model = MultiGraphModel{NegBinEdges{MeanScale},intT,floatT,COV,COF,PAR}(
        model.propensity,
        model.coefficient,
        model.observed,
        model.expected,
        model.covariate,
        new_parameters
    )
end

# Case: NegBinEdges, mean-dispersion, no covariates
function update!(::NegBinEdges{MeanDispersion}, ::Nothing, model, buffers)
    p = model.propensity
    m = length(p)
    x = model.observed
    a = model.parameters.dispersion
    mu = model.expected

    sum_x = sum!(buffers.sum_x, x)
    old_p = copyto!(buffers.old_p, p)
    storage = buffers.threads_buffer

    # Update propensities.
    Threads.@threads for j in 1:m
        t = Threads.threadid()
        storage[t] = 0
        for i in 1:m
            # Use symmetry x[i,j] = x[j,i].
            if i == j continue end
            storage[t] += (a*x[i,j]+1) / (a*mu[i,j]+1) * old_p[i]
        end
        p[j] = sum_x[j] / storage[t]
    end
    update_expectations!(model)

    # Update dispersion parameter, a.
    A, B = 0.0, 0.0
    for j in 1:m, i in 1:m
        if i == j continue end
        for k in 0:(x[i,j]-1)
            B -= inv(a) / (inv(a) + k)
        end
        pi_ij = a*mu[i,j] / (a*mu[i,j]+1)
        A -= x[i,j]
        B -= inv(a) * log1p(-pi_ij) + (x[i,j] + inv(a))*pi_ij
    end
    a = a * (A / B)

    update_expectations!(model)
    new_parameters = (scale=inv(a), dispersion=a)
    intT, floatT, COV, COF, PAR = eltype(x), eltype(mu), typeof(model.covariate), typeof(model.coefficient), typeof(new_parameters)
    return model = MultiGraphModel{NegBinEdges{MeanDispersion},intT,floatT,COV,COF,PAR}(
        model.propensity,
        model.coefficient,
        model.observed,
        model.expected,
        model.covariate,
        new_parameters
    )
end

end # module
