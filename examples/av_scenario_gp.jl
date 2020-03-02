using ExprRules
using ExprOptimization
using Distributions
include("../LTLSampling.jl")
include("plot_trajectories.jl")

include("av_simulator.jl")

# Define the Simulator and action space
N = 20
x = collect(1.:N)
sim = AVSimulator()
car0 = Agent([-35,0], [11.17, 0])
peds0 = [Agent([0,-4], [0,1.5])]

aσ = 1
xσ = 1
vσ = 1

minmaxbound(x)  = [invlogcdf(Normal(0, x), log(0.01)), invlogcdf(Normal(0, x), log(0.99))]
A = ActionSpace(
    :ax => [-2,2],
    :ay => [-2,2],
    :nx => [-1,1],
    :ny => [-1,1],
    :nvx => [-2,2],
    :nvy => [-2,2])



sek(x,xp, l=2, σ2 = 1) = σ2*exp(-(x-xp)^2/(2*l^2))

probability_eval_distributions= Dict(
    :ax => MvNormal(zeros(length(x)), K(x, x, sek)),
    :ay => MvNormal(zeros(length(x)), K(x, x, sek)),
    :nx => MvNormal(zeros(length(x)), xσ*Matrix{Float64}(I, N, N)),
    :ny => MvNormal(zeros(length(x)), xσ*Matrix{Float64}(I, N, N)),
    :nvx => MvNormal(zeros(length(x)), vσ*Matrix{Float64}(I, N, N)),
    :nvy => MvNormal(zeros(length(x)), vσ*Matrix{Float64}(I, N, N))
    )

is_distributions= Dict(
    :ax => MvNormal(zeros(length(x)), 2*K(x, x, sek)),
    :ay => MvNormal(zeros(length(x)), 2*K(x, x, sek)),
    :nx => MvNormal(zeros(length(x)), 2*xσ*Matrix{Float64}(I, N, N)),
    :ny => MvNormal(zeros(length(x)), 2*xσ*Matrix{Float64}(I, N, N)),
    :nvx => MvNormal(zeros(length(x)), 2*vσ*Matrix{Float64}(I, N, N)),
    :nvy => MvNormal(zeros(length(x)), 2*vσ*Matrix{Float64}(I, N, N))
    )


sampling_distributions = Dict{Symbol, Function}(
            :ax => get_constrained_gp_dist(sek),
            :ay => get_constrained_gp_dist(sek),
            :nx => (x,l,u,neq) -> iid_samples(x,l,u,neq,dist=(l,u) -> TruncatedNormal(0,xσ,l,u)),
            :ny => (x,l,u,neq) -> iid_samples(x,l,u,neq,dist=(l,u) -> TruncatedNormal(0,xσ,l,u)),
            :nvx => (x,l,u,neq) -> iid_samples(x,l,u,neq,dist=(l,u) -> TruncatedNormal(0,vσ,l,u)),
            :nvy => (x,l,u,neq) -> iid_samples(x,l,u,neq,dist=(l,u) -> TruncatedNormal(0,vσ,l,u))
            )

# Define the grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = any_between(τ, C, C) | all_between(τ, C, C) # τ is true everywhere before or after C (inclusive)
    C = |(1:20) # A random integer in the domain
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<=")))
    τ = _(sample_sym_comparison(A, Symbol(".>=")))
    τ = _(sample_sym_comparison(A, Symbol(".==")))
end

ex = get_executable(rand(RuleNode, grammar, :R), grammar)

# Define the loss function as the negative of a monte-carlo sampling of the reward
function mc_loss(tree::RuleNode, grammar::Grammar, complexity_param = 0)
    ex = get_executable(tree, grammar)
    trials = 10
    total_loss = 0
    for i=1:trials
        actions = []
        try
            actions = sample_series(ex, A, x, sampling_distributions)
        catch e
            if isa(e, InvalidExpression)
                return 1e9
            else
                println("uncaught error, ", e, " on expr: ", ex)
                error("Uncaught error, ", e)
            end
        end

        ts = create_actions(actions[:ax], actions[:ay], actions[:nx], actions[:ny], actions[:nvx], actions[:nvy])
        total_loss += simulate(sim, ts, car0, peds0, probability_eval_distributions)[1]
    end

    total_loss/trials + complexity_param * count_nodes(tree)
end

gp = GeneticProgram(1000,30,10,0.3,0.3,0.4)
results = optimize(gp, grammar, :R, mc_loss, verbose = true)

println("loss: ", results.loss, " : ", results.expr)

# Show an example of the failure in interactive format
actions = sample_series(results.expr, A, x, sampling_distributions)

r, veh, ped = simulate(sim, create_actions(actions[:ax], actions[:ay], actions[:nx], actions[:ny], actions[:nvx], actions[:nvy]), car0, peds0, probability_eval_distributions)
plot_traj(veh, ped)


# TODO:  Convert to scene
ped_noise = [Agent(ped[i][1].pos .+ (actions[:nx][i], actions[:ny][i]), ped[i][1].vel .+ (actions[:nvx][i], actions[:nvy][i])) for i=1:length(ped)]


for i=1:length(ped)
    res = autoviz_traj(veh[i], ped[i][1], ped_noise[i])
    write_to_svg(res, string("failure_res2_frame", i, ".svg"))
end



# Compute and compare probability of action
gp_failure_avg_prob = []
is_failure_avg_prob = []
for i=1:100
    # print("i=$i, GP....")
    # Fill the average probability of the GP expression
    while true
        a = sample_series(results.expr, A, [1.:N...], sampling_distributions)
        aseq = create_actions(actions[:ax], actions[:ay], actions[:nx], actions[:ny], actions[:nvx], actions[:nvy])
        r, veh, ped = simulate(sim, aseq, car0, peds0, probability_eval_distributions)
        prob = -r / length(aseq)
        if r<0
            push!(gp_failure_avg_prob, prob)
            break
        end
    end
    # println("Is...")
    timeout = 1
    thresh = 1e3
    while true && timeout < thresh
        timeout += 1
        aseq = create_actions(rand(is_distributions[:ax]), rand(is_distributions[:ay]), rand(is_distributions[:nx]), rand(is_distributions[:ny]), rand(is_distributions[:nvx]), rand(is_distributions[:nvy]))
        r, veh, ped = simulate(sim, aseq, car0, peds0, probability_eval_distributions)
        prob = -r / length(aseq)
        if r<0
            push!(is_failure_avg_prob, prob)
            break
        end
    end
    if timeout >= thresh
        println("importance sampling timeout")
    end
end

println("Genetic Programmed expression average action probability: ", mean(gp_failure_avg_prob), " +/- ", std(gp_failure_avg_prob))
println("Uniform IS average action probability: ", mean(is_failure_avg_prob), " +/- ", std(is_failure_avg_prob))

# Fraction of failures (per 100)
Nsamples = 100
Ntrials = 5

function run_trials_gp(expr, Nsamps, Ntrials)
    results = []
    for i=1:Ntrials
        tot_r = 0
        for i=1:Nsamples
            a = sample_series(expr, A, [1.:N...], sampling_distributions)
            aseq = create_actions(actions[:ax], actions[:ay], actions[:nx], actions[:ny], actions[:nvx], actions[:nvy])
            r, veh, ped = simulate(sim, aseq, car0, peds0, probability_eval_distributions)
            tot_r += (r < 0)
        end
        push!(results, tot_r)
    end
    mean(results), std(results)
end

function run_trials_is(Nsamps, Ntrials)
    results = []
    for i=1:Ntrials
        tot_r = 0
        for i=1:Nsamples
            aseq = create_actions(rand(is_distributions[:ax]), rand(is_distributions[:ay]), rand(is_distributions[:nx]), rand(is_distributions[:ny]), rand(is_distributions[:nvx]), rand(is_distributions[:nvy]))
            r, veh, ped = simulate(sim, aseq, car0, peds0, probability_eval_distributions)
            tot_r += (r < 0)
        end
        push!(results, tot_r)
    end
    mean(results), std(results)
end

aseq = create_actions(rand(is_distributions[:ax]), rand(is_distributions[:ay]), rand(is_distributions[:nx]), rand(is_distributions[:ny]), rand(is_distributions[:nvx]), rand(is_distributions[:nvy]))


#nominal behavior
aseq = create_actions(zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N))
r, veh, ped = simulate(sim, aseq, car0, peds0, probability_eval_distributions)


plot_traj(veh, ped)



# run_trials(mdp, FunctionPolicy((s)->random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
mean_gp, std_gp = run_trials_gp(results.expr, Nsamples, Ntrials)
mean_is, std_is  = run_trials_is(Nsamples, Ntrials)
println("gp mean: ", mean_gp, " gp_std: ", std_gp)
println("is mean: ", mean_is, " is_std: ", std_is)

