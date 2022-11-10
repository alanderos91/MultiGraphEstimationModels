# MultiGraphEstimationModels.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/alanderos91/MultiGraphEstimationModels.jl.svg?branch=master)](https://travis-ci.com/alanderos91/MultiGraphEstimationModels.jl)
[![codecov.io](http://codecov.io/github/alanderos91/MultiGraphEstimationModels.jl/coverage.svg?branch=master)](http://codecov.io/github/alanderos91/MultiGraphEstimationModels.jl?branch=master)
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alanderos91.github.io/MultiGraphEstimationModels.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://alanderos91.github.io/MultiGraphEstimationModels.jl/dev)
-->

# Examples

### Poisson Edges

```julia
using MultiGraphEstimationModels, Statistics
MG = MultiGraphEstimationModels

# Simulate a Poisson multigraph with 10 nodes.
ground_truth = simulate_propensity_model(PoissonEdges(), 10, seed=1234)
#
# MultiGraphModel{Int64,Float64}:
#  - distribution: PoissonEdges
#  - nodes: 10
#  - covariates: 0
#

# See the simulated data.
ground_truth.observed
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

# Fit a Poisson model to the data.
fitted = MG.fit(PoissonEdges(), ground_truth.observed; maxiter=10^3, tolerance=1e-6);
# ┌ Info: Converged after 13 iterations.
# │   loglikelihood = -251.21288265649747
# └   initial = -1188.9424092513646

# MSE of propensity estimates with respect to ground truth.
mean( (fitted.propensity .- ground_truth.propensity) .^ 2 )
# 0.12911537825681738

# MSE of expected counts with respect to ground truth.
mean( (fitted.expected .- ground_truth.expected) .^ 2 )
# 8.650578286649598
```