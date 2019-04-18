using ExprRules
using ExprOptimization
using Distributions
using DataStructures

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

a_space = OrderedDict{Symbol, Array{Float64,1}}(:x => [-1, 1], :y => [0,0])

uniform_sample(vals) = (vals[2] == vals[1]) ? vals[1] : rand(Uniform(vals...))
sample_action(a_space) = [uniform_sample(vals) for vals in values(a_space)]

function valid_ation_space(a_space)

function constrain_action_space(a_space, constraints)

end
a_space2 = deepcopy(a_space)
a_space2[:x][1] = -0.5
sample_action(a_space)


rn = rand(RuleNode, grammar, :R, 3)
ex = get_executable(rn, grammar)
string(ex.args[2].args[2], "_min")
eval_conditional_tree(ex, true, 5)

