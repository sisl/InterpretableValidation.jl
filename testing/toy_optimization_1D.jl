using ExprOptimization
using ExprRules
using .LTLSampling

# Define the action space
A = ActionSpace(:x => [0,1])

# Define the grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<="))) # Sample a random less than comparison
    τ = _(sample_sym_comparison(A, Symbol(".>="))) # Sample a random greater than comparison
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end

# Define the target rule node:
target_expr(x) = all(x .> 0.75) && any(x .> 0.95) && any(x .<0.8)


# Define the loss function
function loss(rn::RuleNode, grammar::Grammar)
    ex = get_executable(rn, grammar)
    trials = 10
    total_loss = 0
    for i=1:trials
        actions = []
        try
            actions = sample_series(ex, A, 1:10, iid_samples)
        catch e
            return 1e9
        end

        time_series = actions[:x]
        total_loss -= target_expr(time_series)
    end

    total_loss/trials
end


p = GeneticProgram(1000,20,6,0.3,0.3,0.4)
results_gp = optimize(p, grammar, :R, loss, verbose = true)

println("loss: ", results_gp.loss, " expressions: ", results_gp.expr)

