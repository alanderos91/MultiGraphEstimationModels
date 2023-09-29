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

# Case: PoissonEdges, undirected, no covariates
function __allocate_buffers__(::PoissonEdges, p, ::Nothing)
    buffers = (;
        sum_Z=zeros(Int, length(p)),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:1],
    )
    return buffers
end

# Case: NegBinEdges, undirected, no covariates
function __allocate_buffers__(::NegBinEdges, p, ::Nothing)
    buffers = (;
        sum_Z=zeros(Int, length(p)),
        old_p=similar(p),
        accumulator=[zeros(Threads.nthreads()) for _ in 1:2],
    )
    return buffers
end

# Case: PoissonEdges, undirected, with covariates
function __allocate_buffers__(::PoissonEdges, p, X::AbstractMatrix)
    nnodes = length(p)
    ncovars = size(X, 1)
    nt = Threads.nthreads()
    buffers = (;
        sum_Z=zeros(Int, nnodes),
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

# Case: NegBinEdges, undirected, with covariates
function __allocate_buffers__(::NegBinEdges, p, X::AbstractMatrix)
    nnodes = length(p)
    ncovars = size(X, 1)
    buffers = (;
        sum_Z=zeros(Int, length(p)),
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
