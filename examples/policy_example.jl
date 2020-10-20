using InterpretableValidation
using Distributions
using ExprRules
using Random
using POMDPModels
using POMDPs
using POMDPSimulators
using POMDPPolicies

mdp = SimpleGridWorld(size = (9,9), rewards = Dict(GWPos(5,5) => 1, GWPos(6,5) => 0), tprob = .7, discount=1)

mvts = MvTimeseriesDistribution(:x => IID(collect(1:1.), Categorical(4)))
s_fn(s) = SymbolTable(:sx => s[1], :sy => s[2])
x_fn(v) = actions(mdp)[v[:x][1]]

# Demonstrate that a certain choice of policy obtains high reward
expr = Meta.parse("[(sx .== 5) .& (sy .<= 5), x .== 1] && [(sx .<= 4) .& (sy .<= 5), x .== 4] && [(sx .>= 6) .& (sy .<= 5), x .== 3]")
p = get_policy(expr, s_fn, x_fn, mvts)
mean([simulate(RolloutSimulator(), mdp, RandomPolicy(mdp)) for i=1:1000])
mean([simulate(RolloutSimulator(), mdp, FunctionPolicy(p)) for i=1:1000])

# Setup the grammar
x_samp_dist = default_comparison_distribution(mvts)
x_comp = default_comparisons(mvts)
s_samp_dist = Dict{Symbol, Distribution}(:sx => Categorical(9), :sy => Categorical(9))
all_syms = [Symbol(".<="), Symbol(".=="), Symbol(".>=")]
s_comp = Dict(:sx => all_syms, :sy => all_syms)
g = create_policy_grammar(1, x_samp_dist, x_comp, s_samp_dist, s_comp)

# Setup the loss function
lf = policy_loss_fn(mdp, s_fn, x_fn, mvts)

# optimize
optimize(()->nothing, mvts, Npop = 1000, Niter = 30, loss = lf, grammar = g, max_depth = 5)


