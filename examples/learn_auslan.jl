using .LTLSampling
using Statistics
using ExprRules
using ExprOptimization
using Plots
include("auslan_utils.jl")

auslan_sign = "you" # Define a sign of interest
hand = rcols # Define the hand you want

# Load in all examples for the sign of my choice
data = get_data(auslan_sign, [:thumb_bend, :forefinger_bend], hand)
plot(data[:forefinger_bend])


# Determine the action space
A =ActionSpace(
    :thumb_bend => [0,1],
    :forefinger_bend => [0,1],
    # :middlefinger_bend => [0,1],
    # :ringfinger_bend => [0,1],
    # :littlefinger_bend => [0,1],
    # :roll => [-0.5, 0.5],
    # :pitch => [-0.5, 0.5],
    # :yaw => [-0.5, 0.5],
    # :x => [extrema(data[:x][:])...],
    # :y => [extrema(data[:y][:])...],
    # :z => [extrema(data[:z][:])...]
    )


# Build a gaussian process with the correct kernel
k(x,xp, l=2, σ2 = 0.2) = σ2*exp(-(x-xp)^2/(2*l^2))
gp_dist = get_constrained_gp_dist(k)

# Write loss function that returns average l2 loss of an expression of a similar size
time_series_loss(data, sample) = mean(sum((data .- sample).^2, dims = 1))

function loss(ex, grammar::Grammar)
    # ex = get_executable(rn, grammar)
    trials = 10
    N = 15
    total_loss = 0

    for i=1:trials
        actions = []
        try
            actions = sample_series(ex, A, [1.:15...], gp_dist)
        catch e
            return 1e9
        end

        len = length(keys(actions))
        for a in keys(actions)
            total_loss += time_series_loss(data[a], actions[a])
        end
        total_loss /= len
    end

    total_loss/trials
end

# Define the grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all_before(τ, C) | all_after(τ, C) # τ is true everywhere before or after C (inclusive)
    R = all_between(τ, C, C)
    C = _(rand(1:15)) # A random integer in the domain
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<="))) # Sample a random less than comparison
    τ = _(sample_sym_comparison(A, Symbol(".>="))) # Sample a random greater than comparisonq
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end

# test out the losses
rn = rand(RuleNode, grammar, :R)
ex = get_executable(rn, grammar)

p = plot()
for i=1:10
    actions = sample_series(ex, A, [1.:15...], gp_dist)

    plot!(actions[:forefinger_bend], label="")
end
display(p)


loss(rn, grammar)

# setup the optimization and run
p = GeneticProgram(100,10,6,0.3,0.3,0.4)
results_gp = optimize(p, grammar, :R, loss, verbose = true)

a = sample_series(results_gp.expr, A, [1.:15...], gp_dist)
plot(a[:thumb_bend])








