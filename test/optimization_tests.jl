using InterpretableValidation
using Test
using Distributions
using Random
using ExprRules
using POMDPs
using POMDPSimulators
using POMDPModels
using POMDPPolicies

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
ev(t::Dict{Symbol, Array}) = sum(sum.(values(t)))
ev(rand(mvts))
rn = rand(RuleNode, g, :R, 3)
loss = loss_fn(ev, mvts)
@test loss(rn, g) > 0

results = optimize((x) -> rand(), mvts, Npop=10, Niter=3, verbose = false)
@test results.loss > 0

# Sample until we have a feasible rule
max_tries, trial = 1000, 1
while true
    trial > max_tries && break
    global rn = rand(RuleNode, g, :R, 3)
    constraints = sample_constraints(get_executable(rn, g), N_pts(mvts), Random.GLOBAL_RNG)
    mvts2 = deepcopy(mvts)
    constrain_timeseries!(mvts2, constraints)
    isfeasible(mvts2) && break
end
ex = get_executable(rn, g)
constraints = sample_constraints(ex, N_pts(mvts), Random.GLOBAL_RNG)
mvts2 = deepcopy(mvts)
constrain_timeseries!(mvts2, constraints)
@test isfeasible(mvts2)

loss = loss_fn((x) -> throw(error("itsaerror")), mvts)
@test_throws ErrorException loss(rn, g)

loss = loss_fn((x) -> throw(InfeasibleConstraint("itsaerror")), mvts)
@test loss(rn, g) >= 1e9



mdp = SimpleGridWorld(tprob = 1)
hist = simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), GWPos(3,3))

## Test the playback simulator
playback = PlaybackPolicy(collect(action_hist(hist)), RandomPolicy(mdp))
@test all(playback.actions .== action_hist(hist))
@test playback.backup_policy isa RandomPolicy
@test playback.i == 1

hist2 = simulate(HistoryRecorder(), mdp, playback, GWPos(3,3))
@test hist == hist2

action(playback, GWPos(3,3))

t, fn = discrete_action_mdp(mdp, 1000, use_prob = false)
@test t[:a].timeseries_distribution.distribution.p == 0.25*ones(4)
@test length(t) == 1
@test collect(keys(t))[1] == :a
@test all([t[:a].feasible[i] == [1,2,3,4] for i=1:100])
d = rand(t)
lnp = logpdf(t, d)
fn(d)

aprob = Dict(:up =>0.1, :down => 0.1, :left =>0.1, :right=>0.7)
t, fn = discrete_action_mdp(mdp, 1000, use_prob = true, action_probability = (mdp, a) -> aprob[a])
@test t[:a].timeseries_distribution.distribution.p == [0.1, 0.1, 0.1, 0.7]

h = sample_history(Meta.parse("any(a .== 2)"), t, mdp)
@test h isa SimHistory



