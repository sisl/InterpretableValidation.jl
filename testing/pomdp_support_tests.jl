using InterpretableValidation
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using Test


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

t, fn = discrete_action_mdp(mdp, 1000)
@test length(t) == 1
@test collect(keys(t))[1] == :a
@test all([t[:a].feasible[i] == [1,2,3,4] for i=1:100])

d = rand(t)
fn(d)

