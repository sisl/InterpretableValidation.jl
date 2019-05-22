using ExprOptimization
using Statistics
using CSV
using DataFrames
include("setup_av_scenario.jl")
include("plot_trajectories.jl")
using .LTLSampling

function analyze_algorithm_performance(p, name, loss_fn, prune_result; trials = 16)
    expressions, losses, num_nodes, figures = [], [], [], []
    for i = 1:trials
        println("Trial ", i)
        results = optimize(p, grammar, :R, loss_fn, verbose = true)
        tree = results.tree
        if prune_result
            tree = prune_unused_nodes(tree, grammar, loss_fn, 500.)
        end

        loss = loss_fn(tree, grammar)
        expr = get_executable(tree, grammar)
        push!(figures, plot_examples(expr, 1, "Trial $i"))
        push!(expressions, string(expr))
        push!(losses, loss)
        push!(num_nodes, count_nodes(tree))
    end

    loss_mean, loss_std = mean(losses), std(losses)
    nnodes_mean, nnodes_std = mean(num_nodes), std(num_nodes)
    savefig(plot(figures..., size = (2400, 2400)), string(name, "_figures.pdf"))
    CSV.write(string(name, "_results.csv"), DataFrame(Dict("expression" => expressions, "loss" =>losses)))

    return loss_mean, loss_std, nnodes_mean, nnodes_std
end

gp = GeneticProgram(1000,30,10,0.3,0.3,0.4)
# losses = [ mc_loss, (tree::RuleNode, grammar::Grammar) -> mc_loss(tree, grammar, 20),  mc_loss, (tree::RuleNode, grammar::Grammar) -> mc_loss(tree, grammar, 20)]
losses = [mc_loss]
prune_result = [true]
# prune_result = [false, false, true, true]
pnames = ["GP+prune"]
# pnames = ["GP", "GP+Penalty", "GP+Pruning", "GP+Penalty+Pruning"]

np = length(losses)
loss_means, loss_stds = [], []
nnodes_means, nnodes_stds = [], []
for i in 1:np
    loss_mean, loss_std, nnodes_mean, nnodes_std = analyze_algorithm_performance(gp, pnames[i], losses[i], prune_result[i], trials = 1)
    push!(loss_means, loss_mean)
    push!(loss_stds, loss_std)
    push!(nnodes_means, nnodes_mean)
    push!(nnodes_stds, nnodes_std)
end

# save and plot the results
d = Dict("Average Loss" => loss_means, "Std of Loss" => loss_stds, "Average Number of Nodes" => nnodes_means, "Std of Number of Nodes" => nnodes_stds)

CSV.write("losses_and_complexity_comparison.csv", DataFrame(d))


scatter(-loss_means, xticks = (1:np, pnames), xrotation = 90, yerr = loss_stds, xlabel = "Algorithm", ylabel="Reward", title = "Average Reward over 16 Trials", label="")
savefig("reward_comparison.pdf")

scatter(nnodes_means, xticks = (1:np, pnames), xrotation = 90, yerr = nnodes_stds, xlabel = "Algorithm", ylabel="Number of Nodes in Expression", title = "Average Tree Complexity over 16 Trials", label="")
savefig("complexity_comparison.pdf")

expr = Meta.parse("all(ny .== -0.53) && all(ay .== -0.51)")
plot_examples(expr, 1, "Trial 1")

plot!(size = (700,500), xlims = (-20, 0))

savefig("uber")

