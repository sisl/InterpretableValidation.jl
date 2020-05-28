using InterpretableValidation
using Distributions

x = collect(1:10.)
mvts = MvTimeseriesDistribution(:x => IID(x, Uniform()), :y => IID(x, Normal()), :z => IID(x, Categorical(5)))

function target_expr(data::Dict{Symbol, Array{Float64}})
    truth_vals = ((data[:x] .> 0.75) .& (data[:y] .< -0.75)) .| ((data[:x] .< 0.25) .& (data[:y] .> -0.25))
    truth_vals[1] = truth_vals[1] &&  (data[:z][1] == 5)
    return (length(truth_vals) - sum(truth_vals)) / length(truth_vals)
end

results = optimize(target_expr, mvts)
println("loss: ", results.loss, " expressions: ", results.expr)

