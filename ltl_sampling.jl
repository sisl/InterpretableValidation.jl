using ExprRules
using ExprOptimization

include("inverse_logic.jl")

# Define the LTL grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all(x < c) #| all(x > c) | all(x = c) # This is required so that R can terminate in 1 step if needed
    R = any(x < c) | any(x > c) | any(x == c) # This is required so that R can terminate in 1 step if needed
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = (x < c) | (x > c) | (x == c)# Less than operation
end



rn = rand(RuleNode, grammar, :R, 3)
ex = get_executable(rn, grammar)
ex.args[2].args
eval_conditional_tree(ex, true, 5)

