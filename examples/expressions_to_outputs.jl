using ExprRules
using Plots; gr()
using .LTLSampling

grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all_before(τ, K) | all_after(τ, K)
    K = 12
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = (sym .<= c) | (sym .>= c) | (sym .== c)# Less than operation
    c = 0.01 | 0.25 | 0.5 | 0.75 | 0.99 # Constants for comparison
    sym = x | y
end

A1D = ActionSpace(:x => [-1,1])
A2D = ActionSpace(:x => [0,1], :y => [0,1])

N = 20
x = Float64[1:N...]

# 1D iid samples
rn1 = RuleNode(1, [RuleNode(6,[RuleNode(11, [RuleNode(18), RuleNode(15)]), RuleNode(7)]), RuleNode(4, [RuleNode(11, [RuleNode(18), RuleNode(17)])])])
ex1 = get_executable(rn1, grammar)
constrained_series = sample_series(ex1, A1D, 1:N, iid_samples)
a1_series_uniform = sample_series([A1D for i in 1:N],  1:N, iid_samples)
p = plot(a1_series_uniform[:x], ylims = (-1,1), label="Unconstrained Series", size = (600,200), legend=:bottomright)
plot!(constrained_series[:x], label = "Constrained Series")


# 1D GP Example
m(x) = 0
k(x, x′) = exp(-(x-x′)^2 / (2*2^2))

gp = get_constrained_gp_dist(m, k)

leaves = LTLSampling.eval_conditional_tree(ex1, true, N)
constraints = LTLSampling.gen_constraints(leaves, N)
A_series, valid = LTLSampling.gen_action_spaces(A1D, constraints)
lu_neq, sym_list, N = LTLSampling.get_lu_neq(A_series), syms(A_series[1]), length(A_series)
sym = :x
x
sample_constrained_gp(m, k, x, lu_neq[sym].l,  lu_neq[sym].u, n=2, X=[1.,2,3,4], y=[0.,0,0,0])


constrained_series = sample_series(ex1, A1D, x, gp)
a1_series_uniform = sample_series([A1D for i in 1:N],  x, gp)
p = plot(a1_series_uniform[:x], ylims = (-1,1), label="Unconstrained Series", size = (600,200), legend=:bottomright)
plot!(constrained_series[:x], label = "Constrained Series")
println(replace(replace(string(ex1), "&" =>"\\&"), "_" => "\\_"))
display(p)


# 2D iid sample
# r = RuleNode
# rn2 = r(3, [r(6, [r(5,[r(8,[r(15), r(13)]), r(8,[r(16), r(13)])]),r(5,[r(7,[r(15), r(11)]), r(7,[r(16), r(11)])])])])
# ex2 = get_executable(rn2, grammar)
# constrained_series = sample_series(ex2, A2D, N)
# a2_series_uniform = sample_series([A2D for i in 1:N], iid_samples, 1:N, 1)
# p1 = plot(a2_series[:x], ylims = (0,1), title="Unconstrained Series", label="x")
# plot!(a2_series[:y], label="y")
# p2 = plot(constrained_series[:x], ylims = (0,1), label="x", title = string("Series Constrained by: ", replace(string(ex2), "&" =>"\\&")))
# plot!(constrained_series[:y], label="y")
# plot(p1, p2, layout = (2,1))
