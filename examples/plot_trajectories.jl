using Plots
pyplot()
include("av_simulator.jl")

function plot_examples(ex, nsamps, title = nothing, p = nothing)
    if p == nothing
        p = plot(legend = :topleft, size=(600,600))
    end
    if title == nothing
        title = string(ex)
    end
    title!(title)
    for i=1:nsamps
        action_ts = sample_series(ex, A, 1:50, iid_samples, 1)
        actions = create_actions(action_ts[:ax], action_ts[:ay], action_ts[:nx], action_ts[:ny], action_ts[:nvx], action_ts[:nvy])
        rw, car_traj, ped_traj = simulate(sim, actions, car0, peds0, model)
        N = length(car_traj)
        carx = [car_traj[i].pos[1] for i in 1:N]
        cary = [car_traj[i].pos[2] for i in 1:N]
        plot!(carx, cary, marker=:square, ylims = (-10,10), label="Car Position - $i")
        ypts = [-1.4, -1.4, 1.4, 1.4, -1.4] .+ cary[end]
        xpts = [-2.5, 2.5, 2.5, -2.5, -2.5] .+ carx[end]
        plot!(xpts, ypts, label="", color = :black)

        pedx = [ped_traj[i][1].pos[1] for i in 1:N]
        pedy = [ped_traj[i][1].pos[2] for i in 1:N]
        plot!(pedx, pedy, marker=:circ, label="")
    end
    p
end

