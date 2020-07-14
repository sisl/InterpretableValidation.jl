# Interpretable Safety Validation
[![Build Status](https://travis-ci.org/sisl/InterpretableValidation.jl.svg?branch=master)](https://travis-ci.org/sisl/InterpretableValidation.jl) [![Coverage Status](https://coveralls.io/repos/github/sisl/InterpretableValidation.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/InterpretableValidation.jl?branch=master) [![codecov](https://codecov.io/gh/sisl/InterpretableValidation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/InterpretableValidation.jl)

A collection of tools for the interpretable safety validation of autonomous system. See the following paper for a description of the methods: https://arxiv.org/abs/2004.06805

## Usage
See `examples/sample_optimization.jl` for an example of how to use the package.

The steps include
1. Defining a Multivariate time series distribution
2. Define a cost function
3. Call the `optimize` function with the following options

* `eval_fn::Function` - Function that evaluates the cost of a sample from the `MvTimeseriesDistribution`.
* `d::MvTimeseriesDistribution` - The baseline sampling distribution.
* `rng::AbstractRNG` - The random number generator. Default: `Random.GLOBAL_RNG`.
* `loss` - The loss function that takes a RuleNode as input and returns a loss. Best practice is to set the `eval_fn` and use the default here. Default: `loss_fn(eval_fn, d, rng = rng)`.
* `Npop` - Size of the population. Default: `1000`.
* `Niter` - Number of optimization iterations. Default: `30`.
* `max_depth` - The max tree depth for expressions. Default: `10`.
* `opt` -  The optimization program. This option overides the population, iterations and max_depth if set independently. Default: `GeneticProgram(Npop, Niter, max_depth, 0.3, 0.3, 0.4)`
* `comparison_distribution` - The distribution for sampling comparisons like `x < 0.5`. Default:  `default_comparison_distribution(d)`.
* `grammar` - The grammar used to sample expressions. Default: `create_grammar()`.
* `verbose` - Whether or not to print progress. Default: `true`


Maintained by Anthony Corso (acorso@stanford.edu)
