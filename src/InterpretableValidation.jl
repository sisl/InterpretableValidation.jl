module InterpretableValidation
    using Distributions
    using LinearAlgebra
    using SpecialFunctions
    using NLsolve
    using ExprRules
    using Parameters
    using Random

    # mvrandn
    export mvrandn_μ, mvrandn
    include("mvrandn.jl")

    # Timeseries Distributions
    export TimeseriesDistribution, IID, GaussianProcess, N_pts, squared_exp_kernel
    include("timeseries_distributions.jl")

    # Inverse Logic
    export sample_constraints, flex_not,
            all_before, all_before_inv, all_after, all_after_inv, all_between,
            all_between_inv, any_between, any_between_inv,
            and_inv, or_inv, all_inv, any_inv, bitwise_and_inv, bitwise_or_inv
    include("inverse_logic.jl")

    # Constrained Distributions
    export ConstrainedTimeseriesDistribution, isdiscrete, iscontinuous, isfeasible,
           MvTimeseriesDistribution,
           greaterthan!, lessthan!, continuous_equality!, discrete_equality!, constrain_timeseries!
    include("constrained_distributions.jl")

    # Optimization and Grammar
    export loss_fn, sample_comparison, grammar, default_comparison_distribution
    include("optimization.jl")

end

