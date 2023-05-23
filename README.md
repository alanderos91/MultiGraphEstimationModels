# MultiGraphEstimationModels.jl

<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg)
[![Build Status](https://travis-ci.com/alanderos91/MultiGraphEstimationModels.jl.svg?branch=master)](https://travis-ci.com/alanderos91/MultiGraphEstimationModels.jl)
[![codecov.io](http://codecov.io/github/alanderos91/MultiGraphEstimationModels.jl/coverage.svg?branch=master)](http://codecov.io/github/alanderos91/MultiGraphEstimationModels.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alanderos91.github.io/MultiGraphEstimationModels.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://alanderos91.github.io/MultiGraphEstimationModels.jl/dev)
-->

Analyze networks with integer-valued edge weights as multigraphs.

## Examples

### Propensity-based Models

Propensity-based multigraph models impose a distribution on edge weights/counts and model the expectations as a product of propensities, $E[Z_{i,j}] = p_{i} \cdot p_{j}$.
The nonnegative propensities $p_{i}$ capture the likelihood of a node $i$ forming edges with other nodes and thus capture information about associations.

#### Assuming Poisson distribution

```julia
using MultiGraphEstimationModels, Statistics
mGEM = MultiGraphEstimationModels

# simulate under Poisson assumption
model = mGEM.simulate_propensity_model(PoissonEdges(), 10; seed=1234)
# MultiGraphModel{Int64,Float64}:
#   - distribution: PoissonEdges
#   - nodes: 10
#   - covariates: 0

# see the count data/edge weights
model.observed
# 10×10 Matrix{Int64}:
#   0  34  47  37  26  50  4  5  49  52
#  34   0  43  49  32  53  5  3  47  62
#  47  43   0  44  41  62  4  6  63  86
#  37  49  44   0  44  38  7  7  64  60
#  26  32  41  44   0  45  6  5  48  51
#  50  53  62  38  45   0  5  1  57  45
#   4   5   4   7   6   5  0  0   6   8
#   5   3   6   7   5   1  0  0   5   3
#  49  47  63  64  48  57  6  5   0  76
#  52  62  86  60  51  45  8  3  76   0

# fit a model under the Poisson assumption
result = mGEM.fit_model(PoissonEdges(), model.observed; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 13 iterations.
# │   loglikelihood = -251.21288265649738
# └   initial = -1188.942409251364

# inspect results
result.logl_init
# -1188.942409251364

result.logl_final
# result.logl_final

result.iterations
# 13

result.converged
# true

fitted = result.fitted
# MultiGraphModel{Int64,Float64}:
#   - distribution: PoissonEdges
#   - nodes: 10
#   - covariates: 0

# largest error
maximum(abs, model.propensity - fitted.propensity)
# 0.7218133095737995

# mean squared error
mean(abs2, model.propensity - fitted.propensity)
# 0.12911537825681738
```

#### Assuming Negative Binomial distribution

By default, we parameterize the negative binomial in the mean-scale formulation; that is, the option `NegBinEdges()` is equivalent to `NegBinEdges(MeanScale())`.

This means that `r == model.parameters.scale` and `1/r == model.parameters.dispersion`.

```julia
using MultiGraphEstimationModels, Statistics
mGEM = MultiGraphEstimationModels

# simulate under Poisson assumption
model = mGEM.simulate_propensity_model(NegBinEdges(), 100; seed=1234, dispersion=5.0)
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanScale}
#   - nodes: 100
#   - covariates: 0

# fit a model under the Poisson assumption
result = mGEM.fit_model(NegBinEdges(), model.observed; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 10 iterations.
# │   loglikelihood = -316326.5902995308
# └   initial = -423421.73077315575
# ┌ Info: Converged after 179 iterations.
# │   loglikelihood = -29084.793948267252
# └   initial = -30001.971484287835
fitted = result.fitted
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanScale}
#   - nodes: 100
#   - covariates: 0

# largest error
maximum(abs, model.propensity - fitted.propensity)
# 4.419471977156574

# mean squared error
mean(abs2, model.propensity - fitted.propensity)
# 1.8753282099196358

# nuisance parameter
abs(model.parameters.scale - fitted.parameters.scale)
# 0.003129427781723898

abs(model.parameters.dispersion - fitted.parameters.dispersion)
# 0.07703038934089701
```

Alternatively, we have a separate algorithm for the mean-dispersion parameterization. Specify `NegBinEdges(MeanDispersion())` to use it.

Under this parameterization, we have `1/a == model.parameters.scale` and `a == model.parameters.dispersion`.

```julia
using MultiGraphEstimationModels, Statistics
mGEM = MultiGraphEstimationModels

# simulate under Poisson assumption
model = mGEM.simulate_propensity_model(NegBinEdges(MeanDispersion()), 100; seed=1234, dispersion=5.0)
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanDispersion}
#   - nodes: 100
#   - covariates: 0

# fit a model under the Poisson assumption
result = mGEM.fit_model(NegBinEdges(MeanDispersion()), model.observed; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 10 iterations.
# │   loglikelihood = -316326.5902995308
# └   initial = -423421.7307731593
# ┌ Info: Converged after 261 iterations.
# │   loglikelihood = -29084.191535320642
# └   initial = -30001.971484287806
fitted = result.fitted
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanDispersion}
#   - nodes: 100
#   - covariates: 0

# largest error
maximum(abs, model.propensity - fitted.propensity)
# 4.247552841237734

# mean squared error
mean(abs2, model.propensity - fitted.propensity)
# 1.8184036476844974

# nuisance parameter
abs(model.parameters.scale - fitted.parameters.scale)
# 0.0004173056813826237

abs(model.parameters.dispersion - fitted.parameters.dispersion)
# 0.010410919355588355
```

### Covariate-based Models

We incorporate additional node-specific information by modeling propensities as log-linear models of their covariates
$$
p_{i} \sim \exp({x_{i}^{\top}\beta}),
$$
where $x_{i}$ denotes the covariates associated with node $i$ and $\beta$ is a coefficients vector.

#### Assuming Poisson distribution

```julia
using MultiGraphEstimationModels, Statistics
mGEM = MultiGraphEstimationModels

# simulate under Poisson assumption
model = mGEM.simulate_covariate_model(PoissonEdges(), 1000, 10; seed=1234)
# MultiGraphModel{Int64,Float64}:
#   - distribution: PoissonEdges
#   - nodes: 1000
#   - covariates: 10

# fit a model under the Poisson assumption
result = mGEM.fit_model(PoissonEdges(), model.observed, model.covariate; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 6 iterations.
# │   loglikelihood = -1.3836596581445902e6
# └   initial = -1.2406940032355124e7
fitted = result.fitted
# MultiGraphModel{Int64,Float64}:
#   - distribution: PoissonEdges
#   - nodes: 1000
#   - covariates: 10

# largest error
maximum(abs, model.propensity - fitted.propensity)
# 0.07916104137893143

maximum(abs, model.coefficient - fitted.coefficient)
# 0.0008447705725385807

# mean squared error
mean(abs2, model.propensity - fitted.propensity)
# 2.7222043272077668e-5

mean(abs2, model.coefficient - fitted.coefficient)
# 1.4275988042326197e-7
```

#### Assuming Negative Binomial distribution

```julia
using MultiGraphEstimationModels, Statistics
mGEM = MultiGraphEstimationModels

# simulate under Poisson assumption
model = mGEM.simulate_covariate_model(NegBinEdges(), 1000, 10; dispersion=2.0, seed=1234)
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanScale}
#   - nodes: 1000
#   - covariates: 10

# fit a model under the Poisson assumption
result = mGEM.fit_model(NegBinEdges(), model.observed, model.covariate; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 20 iterations.
# │   loglikelihood = -1.5926005004651072e6
# └   initial = -2.309052251536611e6
fitted = result.fitted
# MultiGraphModel{Int64,Float64}:
#   - distribution: NegBinEdges{MeanScale}
#   - nodes: 1000
#   - covariates: 10

# largest error
maximum(abs, model.propensity - fitted.propensity)
# 0.32012244582232796

maximum(abs, model.coefficient - fitted.coefficient)
# 0.0029198473449780493

# mean squared error
mean(abs2, model.propensity - fitted.propensity)
# 0.0005649879300132473

mean(abs2, model.coefficient - fitted.coefficient)
# 2.508108767946127e-6
```
