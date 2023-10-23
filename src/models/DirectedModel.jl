struct DirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT} <: AbstractMultiGraphModel{distT,intT,floatT,covT}
    propensity_out::Vector{floatT}  # propensity of node i to form an edge
    propensity_in::Vector{floatT}   # propensity of node i to receive an edge
    coefficient_out::coeffT         # effect size of covariate j on donor node propensities
    coefficient_in::coeffT          # effect size of covariate j on recepient node propensities
    observed::Matrix{intT}      # edge count data
    expected::Matrix{floatT}    # expected edge counts under statistical model
    covariate::covT             # nodes × covariates design matrix
    parameters::paramT          # additional parameters
end

function DirectedMultiGraphModel(::distT, observed, covariate, parameters::NamedTuple) where distT <: AbstractEdgeDistribution
    # sanity checks
    check_observed_data(observed)
    check_covariates(observed, covariate)

    # allocate data structures
    propensity_out = allocate_propensities(observed)
    propensity_in = allocate_propensities(observed)
    expected = allocate_expected_matrix(observed)
    coefficient_out = allocate_coefficients(covariate)
    coefficient_in = allocate_coefficients(covariate)

    # determine type parameters
    covT = typeof(covariate)
    coeffT = typeof(coefficient_out)
    paramT = typeof(parameters)
    intT = eltype(observed)
    floatT = eltype(propensity_out)

    model = DirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT}(
        propensity_out, propensity_in,
        coefficient_out, coefficient_in,
        observed,
        expected,
        covariate,
        parameters
    )
    update_expectations!(model)
    return model
end

function Base.show(io::IO, model::DirectedMultiGraphModel{distT,intT,floatT}) where {distT,intT,floatT}
    nnodes = length(model.propensity_out)
    ncovar = model.coefficient_out isa Nothing ? 0 : length(model.coefficient_out)

    print(io, "DirectedMultiGraphModel{$(intT),$(floatT)}:")
    print(io, "\n  - distribution: $(distT)")
    print(io, "\n  - nodes: $(nnodes)")
    print(io, "\n  - covariates: $(ncovar)")

    return nothing
end

function update_expectations!(model::DirectedMultiGraphModel, ::Nothing)
    p = model.propensity_out
    q = model.propensity_in
    mu = model.expected
    @batch per=core for j in eachindex(p)
        for i in eachindex(p)
            if i == j continue end
            mu[i,j] = p[i] * q[j]
        end
    end
end

function update_expectations!(model::DirectedMultiGraphModel, ::Any)
    p = model.propensity_out
    q = model.propensity_in
    X = model.covariate
    b_out = model.coefficient_out
    b_in = model.coefficient_in
    mul!(p, transpose(X), b_out)
    mul!(q, transpose(X), b_in)
    @. p = min(625.0, exp(p))
    @. q = min(625.0, exp(q))
    update_expectations!(model, nothing)
end

function remake_model!(model::DirectedMultiGraphModel{distT}, new_params) where distT
    intT, floatT = eltype(model.observed), eltype(model.expected)
    covT, coeffT, paramT = typeof(model.covariate), typeof(model.coefficient_out), typeof(new_params)
    return DirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT}(
        model.propensity_out,
        model.propensity_in,
        model.coefficient_out,
        model.coefficient_in,
        model.observed,
        model.expected,
        model.covariate,
        new_params,
    )
end

function init_buffers!(model::DirectedMultiGraphModel, buffers)
    sum!(buffers.sum_Z_row, model.observed) # sum over columns j
    sum!(buffers.sum_Z_col, transpose(model.observed)) # sum over rows i
end

function init_old_state(model::DirectedMultiGraphModel, ::Nothing)
    return (; p=copy(model.propensity_out), q=copy(model.propensity_in), params=model.parameters)
end

function init_old_state(model::DirectedMultiGraphModel, ::Any)
    return (; p=copy(model.propensity_out), q=copy(model.propensity_in), params=model.parameters, b_out=copy(model.coefficient_out), b_in=copy(model.coefficient_in))
end

function update_old_state!(state, model::DirectedMultiGraphModel, ::Nothing)
    @. state.p = model.propensity_out
    @. state.q = model.propensity_in
    return (; p=state.p, q=state.q, params=model.parameters)
end

function update_old_state!(state, model::DirectedMultiGraphModel, ::Any)
    @. state.p = model.propensity_out
    @. state.q = model.propensity_in
    @. state.b_out = model.coefficient_out
    @. state.b_in = model.coefficient_in
    return (; p=state.p, q=state.q, params=model.parameters, b_out=state.b_out, b_in=state.b_in)
end

function backtrack_to_old_state!(model::DirectedMultiGraphModel, state, ::Nothing)
    @. model.propensity_out = state.p
    @. model.propensity_in = state.q
    model = remake_model!(model, state.params)
    update_expectations!(model)
    return model
end

function backtrack_to_old_state!(model::DirectedMultiGraphModel, state, ::Any)
    @. model.coefficient_out = state.b_out
    @. model.coefficient_in = state.b_in
    model = remake_model!(model, state.params)
    update_expectations!(model)
    return model
end

function __allocate_buffers__(dist, model::DirectedMultiGraphModel)
    __allocate_buffers__(dist, model.propensity_out, model.propensity_in, model.covariate)
end

function init_model(::PoissonEdges, model::DirectedMultiGraphModel)
    m = length(model.propensity_out)
    sum_Z = sum(model.observed)
    model.propensity_out .= sqrt( sum_Z / (m * (m-1)) )
    model.propensity_in .= model.propensity_out
    if !(model.covariate isa Nothing)
        # initialize with rough estimates of propensities under Poisson model without covariates
        result = fit_model(PoissonEdges(), model.observed; directed=true, maxiter=5, verbose=false)
        A = copy(model.covariate)              # LHS
        b = log.(result.fitted.propensity_out) # RHS
        model.coefficient_out .= A' \ b
        b = log.(result.fitted.propensity_in)  # RHS
        model.coefficient_in .= A' \ b
    end
    update_expectations!(model)
    return model
end

# Case: PoissonEdges, no covariates
function update!(dist::PoissonEdges, ::Nothing, model::DirectedMultiGraphModel, buffers)
    __mm_new_propensity!__(dist, model.propensity_out, buffers.old_p, model.propensity_in, buffers.old_q, buffers.sum_Z_row, buffers.sum_Z_col)
    update_expectations!(model)
    return model
end
