struct UndirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT} <: AbstractMultiGraphModel{distT,intT,floatT,covT}
    propensity::Vector{floatT}  # propensity of node i to form an edge
    coefficient::coeffT         # effect size of covariate j on node propensities
    observed::Matrix{intT}      # edge count data
    expected::Matrix{floatT}    # expected edge counts under statistical model
    covariate::covT             # nodes × covariates design matrix
    parameters::paramT          # additional parameters
end

function UndirectedMultiGraphModel(::distT, observed, covariate, parameters::NamedTuple) where distT <: AbstractEdgeDistribution
    # sanity checks
    check_observed_data(observed)
    check_covariates(observed, covariate)

    # allocate data structures
    propensity = allocate_propensities(observed)
    expected = allocate_expected_matrix(observed)
    coefficient = allocate_coefficients(covariate)

    # determine type parameters
    covT = typeof(covariate)
    coeffT = typeof(coefficient)
    paramT = typeof(parameters)
    intT = eltype(observed)
    floatT = eltype(propensity)

    model = UndirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT}(propensity, coefficient, observed, expected, covariate, parameters)
    update_expectations!(model)
    return model
end

function Base.show(io::IO, model::UndirectedMultiGraphModel{distT,intT,floatT}) where {distT,intT,floatT}
    nnodes = length(model.propensity)
    ncovar = model.coefficient isa Nothing ? 0 : length(model.coefficient)

    print(io, "UndirectedMultiGraphModel{$(intT),$(floatT)}:")
    print(io, "\n  - distribution: $(distT)")
    print(io, "\n  - nodes: $(nnodes)")
    print(io, "\n  - covariates: $(ncovar)")

    return nothing
end

function update_expectations!(model::UndirectedMultiGraphModel, ::Nothing)
    p = model.propensity
    mu = model.expected
    @batch per=core for j in eachindex(p)
        for i in eachindex(p)
            if i == j continue end
            mu[i,j] = p[i] * p[j]
        end
    end
end

function update_expectations!(model::UndirectedMultiGraphModel, ::Any)
    p = model.propensity
    X = model.covariate
    b = model.coefficient
    mul!(p, transpose(X), b)
    @. p = min(625.0, exp(p))
    update_expectations!(model, nothing)
end

function remake_model!(model::UndirectedMultiGraphModel{distT}, new_params) where distT
    intT, floatT = eltype(model.observed), eltype(model.expected)
    covT, coeffT, paramT = typeof(model.covariate), typeof(model.coefficient), typeof(new_params)
    return UndirectedMultiGraphModel{distT,intT,floatT,covT,coeffT,paramT}(
        model.propensity,
        model.coefficient,
        model.observed,
        model.expected,
        model.covariate,
        new_params,
    )
end

function __eval_derivs!__(::PoissonEdges, model::UndirectedMultiGraphModel, buffers)
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

function __eval_derivs!__(::NegBinEdges{MeanScale}, model::UndirectedMultiGraphModel, buffers)
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
    @. tmp1 = M / (M + r) # element-wise operations
    sum!(w, tmp1)
    @. tmp1 = 2*r*($Diagonal(w) + tmp1)
    mul!(transpose(tmp2), tmp1, transpose(X))
    mul!(d2f, X, transpose(tmp2))

    return nothing
end

function __eval_derivs!__(::NegBinEdges{MeanDispersion}, model::UndirectedMultiGraphModel, buffers)
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
    @. tmp1 = a*M/(a*M + 1) # element-wise operations
    sum!(w, tmp1)
    @. tmp1 = 2*r*($Diagonal(w) + tmp1)
    mul!(transpose(tmp2), tmp1, transpose(X))
    mul!(d2f, X, transpose(tmp2))

    return nothing
end

function approx_standard_errors(model::UndirectedMultiGraphModel{distT}, option::Symbol) where distT
    if option == :propensity
        expected_information = propensity_expected_information(distT(), model)
    else option == :coefficient
        if isnothing(model.covariate)
            error("Model does not contain any covariates! Did you mean to estimate standard errors for each `:propensity`?")
        end
        tmp = __allocate_buffers__(model)
        __eval_derivs!__(distT(), model, tmp)
        expected_information = tmp.hessian
    end
    return sqrt.(diag(inv(expected_information)))
end

function propensity_expected_information(::PoissonEdges, model::UndirectedMultiGraphModel)
    p = model.propensity
    m = length(p)
    E = Matrix{Float64}(undef, m, m)
    sum_p = sum(p)
    for j in axes(E, 2), i in axes(E, 1)
        if i == j
            E[i,i] = 2 / p[i] * (sum_p - p[i])
        else
            E[i,j] = 2
        end
    end
    return E
end

function propensity_expected_information(::NegBinEdges{MeanScale}, model::UndirectedMultiGraphModel)
    p = model.propensity
    m = length(p)
    r = model.parameters.scale
    E = Matrix{Float64}(undef, m, m)
    for j in axes(E, 2), i in axes(E, 1)
        if i == j
            sum_p_weighted = 0.0
            for k in axes(E, 1)
                mu_ki = model.expected[k,i]
                pi_ki = mu_ki / (mu_ki + r)
                sum_p_weighted += p[k] * (1 - pi_ki)
            end
            E[i,i] = 2 * r / (p[i]^2) * sum_p_weighted
        else
            mu_ij = model.expected[i,j]
            pi_ij = mu_ij / (mu_ij + r)
            E[i,j] = 2 * (1 - pi_ij)
        end
    end
    return E
end

function propensity_expected_information(::NegBinEdges{MeanDispersion}, model::UndirectedMultiGraphModel)
    p = model.propensity
    m = length(p)
    a = model.parameters.dispersion
    E = Matrix{Float64}(undef, m, m)
    for j in axes(E, 2), i in axes(E, 1)
        if i == j
            sum_p_weighted = 0.0
            for k in axes(E, 1)
                mu_ki = model.expected[k,i]
                pi_ki = a*mu_ki / (a*mu_ki + 1)
                sum_p_weighted += p[k] * (1 - pi_ki)
            end
            E[i,i] = 2 / p[i] * sum_p_weighted
        else
            mu_ij = model.expected[i,j]
            pi_ij = a*mu_ij / (a*mu_ij + 1)
            E[i,j] = 2 * (1 - pi_ij)
        end
    end
    return E
end

function init_buffers!(model::UndirectedMultiGraphModel, buffers)
    sum!(buffers.sum_Z, model.observed)
end

function init_old_state(model::UndirectedMultiGraphModel, ::Nothing)
    return (; p=copy(model.propensity), params=model.parameters)
end

function init_old_state(model::UndirectedMultiGraphModel, ::Any)
    return (; p=copy(model.propensity), params=model.parameters, b=copy(model.coefficient))
end

function update_old_state!(state, model::UndirectedMultiGraphModel, ::Nothing)
    @. state.p = model.propensity
    return (; p=state.p, params=model.parameters)
end

function update_old_state!(state, model::UndirectedMultiGraphModel, ::Any)
    @. state.p = model.propensity
    @. state.b = model.coefficient
    return (; p=state.p, params=model.parameters, b=state.b)
end

function backtrack_to_old_state!(model::UndirectedMultiGraphModel, state, ::Nothing)
    @. model.propensity = state.p
    model = remake_model!(model, state.params)
    update_expectations!(model)
    return model
end

function backtrack_to_old_state!(model::UndirectedMultiGraphModel, state, ::Any)
    @. model.coefficient = state.b
    model = remake_model!(model, state.params)
    update_expectations!(model)
    return model
end

function __allocate_buffers__(model::UndirectedMultiGraphModel{distT}) where distT
    __allocate_buffers__(distT(), model.propensity, model.covariate)
end

function init_model(::PoissonEdges, model::UndirectedMultiGraphModel)
    m = length(model.propensity)
    sum_Z = sum(model.observed)
    model.propensity .= sqrt( sum_Z / (m * (m-1)) )
    if !(model.covariate isa Nothing)
        # initialize with rough estimates of propensities under Poisson model without covariates
        result = fit_model(PoissonEdges(), model.observed; maxiter=5, verbose=false)
        A = copy(model.covariate)           # LHS
        b = log.(result.fitted.propensity)  # RHS
        model.coefficient .= A' \ b
    end
    update_expectations!(model)
    return model
end

function init_model(::NegBinEdges, model::UndirectedMultiGraphModel)
    if model.covariate isa Nothing
        # initialize with rough estimates under Poisson model
        result = fit_model(PoissonEdges(), model.observed, model.covariate; maxiter=5, verbose=false)
        copyto!(model.propensity, result.fitted.propensity)
    else
        # initialize with rough estimates of propensities under negative binomial model without covariates
        result = fit_model(NegBinEdges(), model.observed; maxiter=5, verbose=false)
        A = copy(model.covariate)           # LHS
        b = log.(result.fitted.propensity)  # RHS
        model.coefficient .= A' \ b
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
        logl = eval_loglikelihood(model, nothing, __allocate_buffers__(model))
        if logl > best_logl
            best_r, best_logl = r, logl
        end
    end
    model = remake_model!(model, (scale=best_r, dispersion=inv(best_r)))
    update_expectations!(model)
    return model
end

# Case: PoissonEdges, no covariates
function update!(dist::PoissonEdges, ::Nothing, model::UndirectedMultiGraphModel, buffers)
    __mm_new_propensity!__(dist, model.propensity, buffers.old_p, buffers.sum_Z)
    update_expectations!(model)
    return model
end

# Case: NegBinEdges, mean-scale, no covariates
function update!(dist::NegBinEdges{MeanScale}, ::Nothing, model, buffers)
    # Update propensities.
    __mm_new_propensity!__(dist, model.propensity, buffers.old_p, model.observed, model.expected, buffers.sum_Z, model.parameters.scale)
    update_expectations!(model)

    # Update scale parameter, r.
    new_r = __mm_new_r_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=new_r, dispersion=inv(new_r))
    return remake_model!(model, new_parameters)
end

# Case: NegBinEdges, mean-dispersion, no covariates
function update!(dist::NegBinEdges{MeanDispersion}, ::Nothing, model, buffers)
    # Update propensities.
    __mm_new_propensity!__(dist, model.propensity, buffers.old_p, model.observed, model.expected, buffers.sum_Z, model.parameters.dispersion)
    update_expectations!(model)

    # Update dispersion parameter, a.
    new_a = __mm_new_a_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=inv(new_a), dispersion=new_a)
    return remake_model!(model, new_parameters)
end

# Case: PoissonEdges, with covariates
function update!(::PoissonEdges, ::AbstractMatrix, model::UndirectedMultiGraphModel, buffers)
    # Update coefficients with Newton's method
    __newton_new_coefficients__(model, buffers)
    update_expectations!(model)

    return model
end

# Case: NegBinEdges, mean-scale, with covariates
function update!(::NegBinEdges{MeanScale}, ::AbstractMatrix, model::UndirectedMultiGraphModel, buffers)
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
function update!(::NegBinEdges{MeanDispersion}, ::AbstractMatrix, model::UndirectedMultiGraphModel, buffers)
    # Update coefficients with Newton's method
    __newton_new_coefficients__(model, buffers)
    update_expectations!(model)

    # Update dispersion parameter, a.
    new_a = __mm_new_a_param__(model, buffers)
    update_expectations!(model)

    new_parameters = (scale=inv(new_a), dispersion=new_a)
    return remake_model!(model, new_parameters)
end
