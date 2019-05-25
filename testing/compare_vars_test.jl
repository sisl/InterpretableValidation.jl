using .LTLSampling
using Plots; gr()


N = 15
sym_data = Dict(:y => rand(N))
expr = Meta.parse("all( x .<= 1 ) && all( x .>= y )")
A = ActionSpace(:x => [0,1])
x = collect(1.:15)

leaves = eval_conditional_tree(expr, true, N)
gen_constraints(leaves, N, sym_data)
res = sample_series(expr, A, x, iid_samples, sym_data; max_trials_for_valid = 10)

plot(res[:x])
plot!(sym_data[:y])

