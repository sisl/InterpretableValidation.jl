using InterpretableValidation
using Test
using Distributions
using Random
using ExprRules

rng = Random.GLOBAL_RNG
x = collect(1:100.)
N = length(x)
uniform = ConstrainedTimeseriesDistribution(IID(x, Uniform(2,3)))
normal = ConstrainedTimeseriesDistribution(IID(x, Normal(0.5, 1.)))
categorical = ConstrainedTimeseriesDistribution(IID(x, Categorical([0.1, 0.3, 0.2, 0.2, 0.2])))
gp = ConstrainedTimeseriesDistribution(GaussianProcess(m = (x) -> 0, k = squared_exp_kernel(l=4), x = x))
mvts = MvTimeseriesDistribution(:x => uniform, :y => normal, :z => categorical, :gp =>gp)

# Test the contents of the default range
comparison_distribution = default_comparison_distribution(mvts)
@test comparison_distribution[:x] == Uniform(2,3)
@test comparison_distribution[:y] == Uniform(0.5-3,0.5+3)
@test comparison_distribution[:z] == Categorical(5)
@test comparison_distribution[:gp] isa Uniform
@test comparison_distribution[:gp].a < -2
@test comparison_distribution[:gp].b > 2

# Sample some random comparisons
for i=1:100
    expr = sample_comparison(comparison_distribution, Random.GLOBAL_RNG)
    @test expr.head == :call
    sym = expr.args[2]
    if sym == :z
        @test expr.args[1] == Symbol(".==")
    else
        @test expr.args[1] in [Symbol(".=="),Symbol(".<="), Symbol(".>=")]
        @test expr.args[3] >= comparison_distribution[sym].a
        @test expr.args[3] <= comparison_distribution[sym].b
    end
    @test sym in keys(comparison_distribution)
end

# Test the grammar rules and grammar sampling
g = create_grammar()
@test g.rules[1] == Meta.parse("R && R")
@test g.rules[5] == Meta.parse("any_between(τ, C, C)")
@test g.iseval[7]
@test g.iseval[10]


# Test the loss function
set_global_grammar_params(N, comparison_distribution, rng)
ev(t::Dict{Symbol, Array{Float64}}) = sum(sum.(values(t)))
ev(rand(mvts))
rn = rand(RuleNode, g, :R, 3)
get_executable(rn, g)
loss = loss_fn(ev, mvts)
@test loss(rn, g) > 0

results = optimize((x) -> rand(), mvts, Npop=10, Niter=3, verbose = false)
@test results.loss > 0

