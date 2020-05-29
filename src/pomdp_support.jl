# Policy for playing back a sequence of actions
mutable struct PlaybackPolicy <: Policy
    actions
    backup_policy::Policy
    i::Int64
end

# Constructor for the PlaybackPolicy
PlaybackPolicy(actions, backup_policy::Policy) = PlaybackPolicy(actions, backup_policy, 1)

# Action selection for the PlaybackPolicy
function POMDPs.action(p::PlaybackPolicy, s)
    a = p.i <= length(p.actions) ? p.actions[p.i] : action(p.backup_policy, s)
    p.i += 1
    a
end

# Function to generate the MvTimeseriesDistribution and eval function for a mdp with discrete actions
function discrete_action_mdp(mdp, N::Int64; backup_policy = RandomPolicy(mdp), rng = Random.GLOBAL_RNG, use_prob = true, action_probability = nothing)
    x = collect(1.:N)
    as = actions(mdp)
    d = (use_prob) ? Categorical([action_probability(mdp, a) for a in as]) : Categorical(length(as))
    t = MvTimeseriesDistribution(:a => IID(x, d))
    f = function eval_fn(d::Dict{Symbol, Array{Float64}})
        as = actions(mdp)[d[:a]]
        r = simulate(RolloutSimulator(rng), mdp, PlaybackPolicy(as, backup_policy))
        (use_prob) ? -log(r) - logpdf(t, d) : -r
    end
    t, f
end

