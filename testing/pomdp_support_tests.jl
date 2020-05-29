using InterpretableValidation
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using Test
using Distributions


mdp = SimpleGridWorld(tprob = 1)
hist = simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), GWPos(3,3))

## Test the playback simulator
playback = PlaybackPolicy(collect(action_hist(hist)), RandomPolicy(mdp))
@test all(playback.actions .== action_hist(hist))
@test playback.backup_policy isa RandomPolicy
@test playback.i == 1

hist2 = simulate(HistoryRecorder(), mdp, playback, GWPos(3,3))
@test hist == hist2

action(playback, GWPos(3,3))

t, fn = discrete_action_mdp(mdp, 1000, use_prob = false)
@test t[:a].timeseries_distribution.distribution.p == 0.25*ones(4)
@test length(t) == 1
@test collect(keys(t))[1] == :a
@test all([t[:a].feasible[i] == [1,2,3,4] for i=1:100])
d = rand(t)
lnp = logpdf(t, d)
fn(d)

aprob = Dict(:up =>0.1, :down => 0.1, :left =>0.1, :right=>0.7)
t, fn = discrete_action_mdp(mdp, 1000, use_prob = true, action_probability = (mdp, a) -> aprob[a])
@test t[:a].timeseries_distribution.distribution.p == [0.1, 0.1, 0.1, 0.7]

