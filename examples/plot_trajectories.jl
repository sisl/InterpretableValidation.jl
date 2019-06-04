using Plots
pyplot()
# include("av_simulator.jl")
include("setup_av_scenario.jl")
using .LTLSampling

function plot_examples(ex, nsamps, title = nothing, p = nothing)
    if p == nothing
        p = plot(legend = :topleft, size=(700,300), xlims=(-19, 3))
    end
    if title == nothing
        title = string(ex)
    end
    title!(title)
    for i=1:nsamps
        action_ts = sample_series(ex, A, 1:50, iid_samples)
        actions = create_actions(action_ts[:ax], action_ts[:ay], action_ts[:nx], action_ts[:ny], action_ts[:nvx], action_ts[:nvy])
        rw, car_traj, ped_traj = simulate(sim, actions, car0, peds0, model)
        N = length(car_traj)
        carx = [car_traj[i].pos[1] for i in 1:N]
        cary = [car_traj[i].pos[2] for i in 1:N]
        plot!(carx, cary, marker=:square, ylims = (-4,4), label="Ego Vehicle" #=Car Position- $i"=#)
        ypts = [-1.4, -1.4, 1.4, 1.4, -1.4] .+ cary[end]
        xpts = [-2.5, 2.5, 2.5, -2.5, -2.5] .+ carx[end]
        plot!(xpts, ypts, label="", color = :black)

        pedx = [ped_traj[i][1].pos[1] for i in 1:N]
        pedy = [ped_traj[i][1].pos[2] for i in 1:N]
        plot!(pedx[1:2:N], pedy[1:2:N], marker=:circ, label="Pedestrian")
    end
    p
end

# ex = Meta.parse("all(((ax .<= 0.1291635133369189) .& (ny .== -0.9742963087069332)) .& (ay .== -0.30090443707542924))")
#
# plot_examples(ex, 1, "")
# savefig("ltl_av_example.pdf")

