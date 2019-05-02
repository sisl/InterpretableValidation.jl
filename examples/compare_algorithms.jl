using ExprOptimization
using LinearAlgebra
using CSV
include("setup_av_scenario.jl")
include("plot_trajectories.jl")

function analyze_algorithm_performance(p, name; trials = 16, loss_fn = generic_loss)
    expressions, losses, num_nodes, figures = [], [], [], []
    for i = 1:trials
        println("Trial ", i)
        results = optimize(p, grammar, :R, loss_fn, verbose = true)
        push!(figures, plot_examples(results.expr, 1, "Trial $i"))
        push!(expressions, string(results.expr))
        push!(losses, results.loss)
        push!(num_nodes, count_nodes(results.tree))
    end

    loss_mean, loss_std = mean(losses), std(losses)
    nnodes_mean, nnodes_std = mean(num_nodes), std(num_nodes)
    savefig(plot(figures..., size = (2400, 2400)), string(name, "_figures.pdf"))
    CSV.write(string(name, "_results.csv"), DataFrame(Dict("expression" => expressions, "loss" =>losses)))

    return loss_mean, loss_std, nnodes_mean, nnodes_std
end

npop = 1000
iterations = 30
gp = GeneticProgram(npop,iterations,10,0.3,0.3,0.4)
ge = GrammaticalEvolution(grammar,:Real,npop,iterations,10,10,6,0.2,0.4,0.4; select_method=GrammaticalEvolutions.TruncationSelection(300))
ce = CrossEntropy(npop,iterations,6,500)
pipe = PIPE(PPT(0.8),npop,iterations,0.2,0.1,0.05,1,0.2,0.6,0.999,6)


programs = [gp, ge, ce, pipe]
pnames = ["Genetic Programming", "Grammatical Evolution", "Cross Entropy", "PIPE"]
np = length(programs)
loss_means, loss_stds = [], []
nnodes_means, nnodes_stds = [], []
for i in 1:np
    loss_mean, loss_std, nnodes_mean, nnodes_std = analyze_algorithm_performance(programs[i], pnames[i])
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










