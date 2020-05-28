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
function discrete_action_mdp(mdp, N::Int64; backup_policy = RandomPolicy(mdp), rng = Random.GLOBAL_RNG, use_prob = true)
    x = collect(1.:N)
    t = MvTimeseriesDistribution(:a => IID(x, Categorical(length(actions(mdp)))))
    f = function eval_fn(d::Dict{Symbol, Array{Float64}})
        lnp =
        as = actions(mdp)[d[:a]]
        r = simulate(RolloutSimulator(rng), mdp, PlaybackPolicy(as, backup_policy))
        (use_prob) ? -log(r) - logpdf(t, d) : -r
    end
    t, f
end

