using ExprRules
using ExprOptimization

include("inverse_logic.jl")

# Define the LTL grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all(x < c) #| all(x > c) | all(x = c) # This is required so that R can terminate in 1 step if needed
    R = any(x < c) | any(x > c) | any(x = c) # This is required so that R can terminate in 1 step if needed
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = (x < c) | (x > c) | (x = c)# Less than operation
end

# Get a list of expressions and outcomes that need to be satisfied for the top level expression to be true
function eval_conditional_tree(expr, cond, N)
    results = []
    eval_conditional_tree(expr, cond, results, N)
    results
end

# This version passes around a list of constraints (expression, values) pairs that need to be satisfied
function eval_conditional_tree(expr, cond, results, N)
    if expr.head == :call && expr.args[1] in [:any, :all]
        inv = bool_inverses[expr.args[1]](cond, N)
        push!(results, [expr.args[2], inv])
        return
    end
    inv = bool_inverses[expr.head](cond)
    for i in 1:length(inv)
        eval_conditional_tree(expr.args[i], inv[i], results, N)
    end
end

rn = rand(RuleNode, grammar, :R, 2)
ex = get_executable(rn, grammar)
ex.head
ex.args[2].head

eval_conditional_tree(ex, true, 5)

