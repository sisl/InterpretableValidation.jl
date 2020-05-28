using InterpretableValidation
using ExprOptimization
using Distributions
using Random
rng = Random.GLOBAL_RNG

x = collect(1:10.)
mvts = MvTimeseriesDistribution(:x => IID(x, Uniform()), :y => IID(x, Normal()), :z => IID(x, Categorical(5)))
N = length(x)
comparison_distribution = default_comparison_distribution(mvts)

function target_expr(data::Dict{Symbol, Array{Float64}})
    truth_vals = ((data[:x] .> 0.75) .& (data[:y] .< -0.75)) .| ((data[:x] .< 0.25) .& (data[:y] .> -0.25))
    truth_vals[1] = truth_vals[1] &&  (data[:z][1] == 5)
    return (length(truth_vals) - sum(truth_vals)) / length(truth_vals)
end
loss = loss_fn(target_expr, mvts)

p = GeneticProgram(1000, 30, 10, 0.3, 0.3, 0.4)
results = optimize(p, grammar(), :R, loss, verbose = true)

println("loss: ", results.loss, " expressions: ", results.expr)

