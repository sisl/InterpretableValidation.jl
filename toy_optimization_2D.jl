using ExprOptimization
using ExprRules
include("LTLSampling.jl")

# Define the action space
A = ActionSpace(:x => [0,1], :y => [-1,0])

# Define the grammar
grammar = @grammar begin
    R = (R && R) #| (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<"))) # Sample a random less than comparison
    τ = _(sample_sym_comparison(A, Symbol(".>"))) # Sample a random greater than comparison
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end


# Define the target rule node:
function target_expr(x, y)
    truth_vals = ((x .> 0.75) .& (y .< -0.75)) .| ((x .< 0.25) .& (y .> -0.25))
    return sum(truth_vals) / length(truth_vals)
end

# Define the loss function
function loss(rn::RuleNode, grammar::Grammar)
    ex = get_executable(rn, grammar)
    trials = 10
    N = 10
    total_loss = 0
    for i in 1:trials
        time_series = Dict()
        try
            time_series = sample_series(ex, A, N)
        catch e
            return 0 # worst possible score if the expression is invalid
        end
        total_loss -= target_expr(time_series[:x], time_series[:y])
    end
    total_loss/trials
end

p = GeneticProgram(1000,30,10,0.3,0.3,0.4)
results_gp = optimize(p, grammar, :R, loss, verbose = true)

