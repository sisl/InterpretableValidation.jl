# using .LTLSampling
using Statistics
using ExprRules
using ExprOptimization
using Plots
using DataStructures
include("auslan_utils.jl")
println("finished loading in resources")

auslan_sign = "you" # Define a sign of interest
hand = rcols # Define the hand you want
vars = collect(keys(hand))

# Load in all examples for the sign of my choice
data = get_data(auslan_sign, vars, hand)
T = size(data[vars[1]], 1)
xx = [1.:T...]
indices = [1:T...]

unary_minus(x) = -x

# Express the full action space
Afull = ActionSpace(
    :thumb_bend => [0,1],
    :forefinger_bend => [0,1],
    :middlefinger_bend => [0,1],
    :ringfinger_bend => [0,1],
    :littlefinger_bend => [0,1],
    :roll => [-0.5, 0.5],
    :pitch => [-0.5, 0.5],
    :yaw => [-0.5, 0.5],
    :x => [extrema(data[:x][:])...],
    :y => [extrema(data[:y][:])...],
    :z => [extrema(data[:z][:])...]
    )

# Build a gaussian process with the correct kernel
k(x,xp, l=2, σ2 = 0.2) = σ2*exp(-(x-xp)^2/(2*l^2))
gp_dist = get_constrained_gp_dist(k)

# Write loss function that returns average l2 loss of an expression of a similar size
time_series_loss(data, sample) = mean(sum((data .- sample).^2, dims = 1))

# Define a dictionary of expressions for each dimension
found_expressions = OrderedDict()
grammar = nothing

# Loop through the action dimensions and solve the optimization problem one at a time
for a in vars
    println("Optimizing expression for ", a)
    known_vals = collect(keys(found_expressions))

    # Define an action space for  the single action
    A = ActionSpace(a => Afull.bounds[a])
    rng = range(A.bounds[a]..., length=10)

    # Define a grammar
    global grammar = Meta.eval(macroexpand(Main, Meta.parse(string("@grammar begin
        R = (R && R)
        R = all(τ) | any(τ) | all_before(τ, C) | all_after(τ, C) | all_between(τ, C, C)
        C = |($indices)
        τ = (τ .& τ)
        τ = ($a .<= G) | ($a .== G) | ($a .>= G)
        G = |($rng)", isempty(known_vals) ? "" : "
        G = H
        H = |($known_vals)
        H = (H + G) | (H*G) | (H - G) ","
    end"))))

    # Define a loss
    function loss(rn::RuleNode, grammar::Grammar)
        ex = get_executable(rn, grammar)
        total_loss, trials = 0, 10

        for i=1:trials
            actions = []
            try
                actions = iterative_sample(a, ex, Afull, xx, gp_dist, found_expressions)[a]
            catch e
                if isa(e, InvalidExpression)
                    return 1e9
                else
                    println("uncaught error, ", e, " on expr: ", ex)
                    error("Uncaught error, ", e)
                end
            end
            total_loss += time_series_loss(data[a], actions)
        end

        total_loss/trials
    end

    # Optimize the function
    p = GeneticProgram(100,10,6,0.3,0.3,0.4)
    println("Solving...")
    results_gp = optimize(p, grammar, :R, loss, verbose = true)

    # store the result
    found_expressions[a] = results_gp.expr

    println("found expression: ", results_gp.expr, " for ", a, " with loss: ", results_gp.loss)
end

grammar

found_expressions

dat = Dict(:forefinger_bend => rand(15), :x => rand(15))

# ex = get_executable(rand(RuleNode, grammar, :R), grammar)
ex = Meta.parse("all((roll .>= forefinger_bend * x) .& (((roll .>= forefinger_bend) .& (roll .>= -0.5)) .& (roll .<= -0.3888888888888889)))")

leaves = eval_conditional_tree(ex, true, 15)
constraints = gen_constraints(leaves, 15, dat)
action_spaces, valid = gen_action_spaces(ActionSpace(Afull, :roll), constraints)

sample_series(ex, ActionSpace(Afull, :roll), xx, gp_dist, dat)

grammar

