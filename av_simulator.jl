using Distances
using StaticArrays
using LinearAlgebra
import Base.+
import Base.-

struct PedestrianModel
    cov_ax::Float64
    cov_ay::Float64
    cov_noise_x::Float64
    cov_noise_y::Float64
    cov_noise_vx::Float64
    cov_noise_vy::Float64
end

PedestrianModel(cov_ax, cov_ay, cov_noise) = PedestrianModel(cov_ax, cov_ay, cov_noise, cov_noise, cov_noise, cov_noise)

to_array(pm::PedestrianModel) = [pm.cov_ax, pm.cov_ay, pm.cov_noise_x, pm.cov_noise_y, pm.cov_noise_vx, pm.cov_noise_vy]

mutable struct Agent
    pos::MVector{2, Float64}
    vel::MVector{2,Float64}
end

to_array(agent::Agent) = [agent.pos..., agent.vel...]

(+)(a1::Agent, a2::Agent) = Agent(a1.pos .+ a2.pos, a1.vel .+ a2.vel)
(-)(a1::Agent, a2::Agent) = Agent(a1.pos .- a2.pos, a1.vel .- a2.vel)

struct OnePedAction
    ped_accel::MVector{2, Float64}
    agent_noise::Agent
end

Action = Array{OnePedAction, 1}

to_array(opa::OnePedAction) = [opa.ped_accel..., to_array(opa.agent_noise)...]

to_array(a::Action) = vcat([to_array(opa) for opa in a]...)

struct AVSimulator
    dt::Float64
    alpha::Float64
    beta::Float64
    v_des::Float64
    delta::Float64
    t_headway::Float64
    a_max::Float64
    s_min::Float64
    d_cmf::Float64
    d_max::Float64
    min_dist::MVector{2,Float64}
end

AVSimulator(dt = 0.1, alpha = 0.85, beta = 0.005, v_des = 11.17, delta = 4.0, t_headway = 1.5, a_max = 3.0, s_min = 4.0, d_cmf = 2.0, d_max = 9.0, min_dist_x = 2.5, min_dist_y = 1.4) = AVSimulator(dt, alpha, beta, v_des, delta, t_headway, a_max, s_min, d_cmf, d_max, MVector{2}([min_dist_x, min_dist_y]))

function M_dist(model, action)
    a_arr = to_array(action)
    return -mahalanobis(a_arr, zeros(size(a_arr)), inv(diagm(0 => to_array(model))))
end

function compute_reward(actions, is_goal, dist_heuristic, model)
    reward = 0
    # Before the end of the episode
    for a in actions
        reward += M_dist(model , a)
    end

    # No crash give the low reward
    if !is_goal
        reward += -10000 - 1000*dist_heuristic
    end
    reward
end

function simulate(sim::AVSimulator, actions::Array{Action, 1}, car::Agent, peds::Array{Agent}, model::PedestrianModel)

    npeds = length(peds)
    @assert npeds == length(actions[1])
    car_accel, car_obs = [0.,0.], [peds[i] - car for i in npeds]
    car_traj, ped_traj, ai = [], [], 1
    for a in actions
        println("action: ", a)
        println("car: ", car)
        println("ped: ", peds[1])
        println("============================================")
        for i in npeds
            update_agent!(peds[i], a[i].ped_accel, sim.dt)
        end

        update_agent!(car, car_accel, sim.dt)

        # take new measurements and noise them
        measurements = [peds[i] + a[i].agent_noise for i in 1:npeds]

        # filter out the noise with an alpha-beta tracker
        car_obs = tracker(sim, car_obs, measurements)

        # select the SUT action for the next timestep
        car_accel[1] = compute_car_accel(sim, car_obs, car)

        push!(car_traj, deepcopy(car))
        push!(ped_traj, deepcopy(peds))

        # check if a crash has occurred. If so return the timestep, otherwise continue
        if is_goal(peds, car, sim.min_dist)
            r = compute_reward(actions[1:ai], true, NaN, model)
            return r, car_traj, ped_traj
        end
        ai += 1
    end

    r = compute_reward(actions, false, closest_dist(peds, car), model)
    r, car_traj, ped_traj
end

function update_agent!(agent, accel, dt)
    agent.pos += dt*agent.vel + 0.5*dt^2*accel
    agent.vel += dt*accel
end

closest_dist(peds, car) = minimum([norm(p.pos - car.pos) for p in peds])

function is_goal(peds, car, min_dist)
    for p in peds
        all(abs.(p.pos - car.pos) .< min_dist) && return true
    end
    return false
end

function tracker(sim, observation_old, measurements)
    npeds = length(observation_old)
    observation = [Agent(zeros(2), zeros(2)) for i in 1:npeds]

    for p in 1:npeds
        observation[p].vel = observation_old[p].vel
        observation[p].pos = observation_old[p].pos + sim.dt * observation_old[p].vel
        residuals = measurements[p].pos - observation[p].pos

        observation[p].pos += sim.alpha .* residuals
        observation[p].vel += (sim.beta / sim.dt) .* residuals
    end

    observation
end

in_road(agent) = agent.pos[2] > -1.5 && agent.pos[2] < 4.5

function compute_car_accel(sim, obs, car)
    v_car = car.vel[1]
    peds_in_road = obs[in_road.(obs)]

    if length(peds_in_road) != 0
        agent_with_miny = peds_in_road[argmin([p.pos[2] for p in peds_in_road])]
        v_oth = agent_with_miny.vel[1]
        s_headway = agent_with_miny.pos[1] - car.pos[1]

        del_v = v_oth - v_car
        s_des = sim.s_min + v_car * sim.t_headway - v_car * del_v / (2 * sqrt(sim.a_max * sim.d_cmf))
        if sim.v_des > 0.0
            v_ratio = v_car / sim.v_des
        else
            v_ratio = 1.0
        end

        a = sim.a_max * (1.0 - v_ratio ^ sim.delta - (s_des / s_headway) ^ 2)
    else
        del_v = sim.v_des - v_car
        a = del_v
    end
    clamp(a, -sim.d_max, sim.a_max)
end

