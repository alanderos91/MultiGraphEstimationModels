module MultiGraphEstimationModels

import Base: show

abstract type AbstractEdgeDistribution end

struct PoissonEdges <: AbstractEdgeDistribution end
struct NegativeBinomialEdges <: AbstractEdgeDistribution end

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

allocate_propensities(observed) = zeros(Float64, size(observed, 1))

allocate_expected_matrix(observed::AbstractMatrix) = zeros(Float64, size(observed))

allocate_coefficients(covariates::AbstractMatrix) = zeros(Float64, size(covariates, 1))
allocate_coefficients(covariates::Nothing) = nothing

struct MultiGraphModel{DIST,intT,floatT,COVARIATE,COEFFICIENT}
    propensity::Vector{floatT}  # propensity of node i to form an edge
    coefficient::COEFFICIENT    # effect size of covariate j on node propensities
    observed::Matrix{intT}      # edge count data
    expected::Matrix{floatT}    # expected edge counts under statistical model
    covariate::COVARIATE        # nodes × covariates design matrix

    function MultiGraphModel(::DIST, observed, covariate) where DIST
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
        intT = eltype(observed)
        floatT = eltype(propensity)

        new{DIST,intT,floatT,COVARIATE,COEFFICIENT}(propensity, coefficient, observed, expected, covariate)
    end
end

MultiGraphModel(edge_distribution, observed_data) = MultiGraphModel(edge_distribution, copy(observed_data), nothing)

MultiGraphModel(edge_distribution, observed_data, covariate_data) = MultiGraphModel(edge_distribution, copy(observed_data), copy(covariate_data))

function Base.show(io::IO, model::MultiGraphModel{DIST,intT,floatT}) where {DIST,intT,floatT}
    nnodes = length(model.propensity)
    ncovar = model.coefficient isa Nothing ? 0 : length(model.coefficient)

    print(io, "MultiGraphModel{$(intT),$(floatT)}:")
    print(io, "\n  - distribution: $(DIST)")
    print(io, "\n  - nodes: $(nnodes)")
    print(io, "\n  - covariates: $(ncovar)")

    return nothing
end

export PoissonEdges, NegativeBinomialEdges, MultiGraphModel

end # module
