using InterpretableValidation
using Test
using Distributions
using Random

## Test construction of ConstrainedTimeseriesDistribution
x = collect(1:100.)
uniform = ConstrainedTimeseriesDistribution(IID(x, Uniform(2,3)))
v = rand(uniform)
@test all(v.>2) && all(v.<3)

normal = ConstrainedTimeseriesDistribution(IID(x, Normal(0.5, 1.)))
normal.lb = uniform.lb
v = rand(normal)
@test all(v.>2)

categorical = ConstrainedTimeseriesDistribution(IID(x, Categorical(5)))
v = rand(categorical)
@test all([v[i] in collect(1:5) for i=1:length(v)])

gp = ConstrainedTimeseriesDistribution(GaussianProcess(m = (x) -> 0, k = squared_exp_kernel(l=4), x = x))
gp.lb = -1*ones(100)
v = rand(gp)
@test all(v .> -1)

@test N_pts(gp) == 100

## Test discrete vs. continuous
@test !isdiscrete(uniform)
@test isdiscrete(categorical)
@test !iscontinuous(categorical)
@test iscontinuous(gp)

## Test feasible check
@test isfeasible(uniform)
@test isfeasible(categorical)

old = categorical.feasible[10]
categorical.feasible[10] = Int64[]
@test !isfeasible(categorical)
categorical.feasible[10] = old

old = uniform.lb[1]
uniform.lb[1]=100
@test !isfeasible(uniform)
uniform.lb[1]=old

## Test construction of MvTimeseriesDistribution
mvts = MvTimeseriesDistribution(:x => uniform, :y =>normal, :z => categorical, :gp =>gp)
@test N_pts(mvts) == 100

v = rand(mvts)
@test length(v) == 4


## Test the application of constraints
uniform.lb = -Inf*ones(100)
uniform.ub = Inf*ones(100)
greaterthan!(uniform, 3., [fill(true, 50)..., fill(false, 50)...])
@test all(uniform.lb[1:50] .== 3.0)
@test all(uniform.ub[51:100] .== 3.0)

greaterthan!(uniform, 2., [fill(true, 50)..., fill(false, 50)...])
@test all(uniform.lb[1:50] .== 3.0)
@test all(uniform.ub[51:100] .== 2.0)

greaterthan!(uniform, 4., [fill(true, 50)..., fill(false, 50)...])
@test all(uniform.lb[1:50] .== 4.0)
@test all(uniform.ub[51:100] .== 2.0)

greaterthan!(uniform, 5., [fill(true, 50)..., fill(:anybool, 50)...])
@test all(uniform.lb[1:50] .== 5.0)
@test all(uniform.ub[51:100] .== 2.0)

lessthan!(uniform, 4., [fill(true, 50)..., fill(false, 50)...])
@test all(uniform.ub[1:50] .== 4.0)
@test all(uniform.lb[51:100] .== 4.0)

uniform.lb = -Inf*ones(100)
uniform.ub = Inf*ones(100)
continuous_equality!(uniform, 3., [fill(true, 50)..., fill(:anybool, 50)...])
@test all(uniform.lb[1:50] .== uniform.ub[1:50])
@test all(rand(uniform)[1:50] .== 3.)

discrete_equality!(categorical, 3, [:anybool, fill(true, 50)..., fill(false, 49)...])
@test categorical.feasible[1] == [1,2,3,4,5]
@test all([categorical.feasible[i] == [3] for i=2:51])
@test all([!(3 in categorical.feasible[i]) for i in 52:100])

## Test constraint on the time series
constrain_timeseries!(mvts, Meta.parse("x .<= 1."), Array{Any}(fill(true, 100)))
@test mvts[:x].ub == ones(100)

constraints = [(Meta.parse("y .<= 1."), Array{Any}(fill(true, 100)))]
constrain_timeseries!(mvts, constraints)
@test mvts[:y].ub == ones(100)


## Test the sampling based on expression
for (key, val) in mvts
    if iscontinuous(val)
        val.lb = -Inf*ones(100)
        val.ub = Inf*ones(100)
    elseif isdiscrete(val)
        val.feasible = fill([1,2,3,4,5], 100)
    end
end
v = Base.rand(Random.GLOBAL_RNG, Meta.parse("all(x .>= 1.) && all(y .>= 1.) && all(z .== 2)"), mvts)
@test all(v[:x] .> 1.)
@test all(v[:y] .> 1.)
@test all(v[:z] .== 2)

@test_throws InfeasibleConstraint Base.rand(Random.GLOBAL_RNG, Meta.parse("all(x .>= 1.) && all(x .<= 0.5)"), mvts)

mvts = MvTimeseriesDistribution(:x => IID(x, Uniform()), :y => IID(x, Normal()), :z => IID(x, Categorical(5)))

function target_expr(data::Dict{Symbol, Array})
    truth_vals = ((data[:x] .> 0.75) .& (data[:y] .< -0.75)) .| ((data[:x] .< 0.25) .& (data[:y] .> -0.25))
    truth_vals[1] = truth_vals[1] &&  (data[:z][1] == 5)
    return (length(truth_vals) - sum(truth_vals)) / length(truth_vals)
end

good_expr = Meta.parse("all(((x .>= 0.75) .& (y .<= -0.75)) .| ((x .<= 0.25) .& (y .>= -0.25)))")
good_vals = rand(Random.GLOBAL_RNG, good_expr, mvts)
@test target_expr(good_vals) <= 0.01


expr = Meta.parse("all_between((x .== 2.881512049363103) .| (x .<= 2.3141385919362447), 59, 14)")
constraints = sample_constraints(expr, 100, Random.GLOBAL_RNG)
mvts = MvTimeseriesDistribution(:x => IID(x, Uniform(2,3)))
good_vals = rand(Random.GLOBAL_RNG, expr, mvts)
x = good_vals[:x]
@test all_between((x .== 2.881512049363103) .| (x .<= 2.3141385919362447), 59, 14)
