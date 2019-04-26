using Plots
using DataFrames
using CSV
include("av_simulator.jl")

agent_traj(x,y,vx,vy) = [Agent(x[i], y[i], vx[i], vy[i]) for i in length(x)]
create_actions(ax, ay, nx, ny, nvx, nvy) = [[OnePedAction([ax[i], ay[i]], Agent([nx[i], ny[i]], [nvx[i], nvy[i]]))] for i in 1:length(ax)]


function get_actions(df, trial)
    trials = df[Symbol("# trial")]
    trial_rows = findall(trials .== trial)
    x_car = df[trial_rows, Symbol(" x_car")]
    y_car = df[trial_rows, Symbol(" y_car")]
    vx_car = df[trial_rows, Symbol(" v_x_car")]
    vy_car = df[trial_rows, Symbol(" v_y_car")]
    x_ped = df[trial_rows, Symbol("x_ped_0")]
    y_ped = df[trial_rows, Symbol("y_ped_0")]
    vx_ped = df[trial_rows, Symbol(" v_x_ped_0")]
    vy_ped = df[trial_rows, Symbol("v_y_ped_0")]
    ax_ped = df[trial_rows, Symbol("a_x_0")]
    ay_ped = df[trial_rows, Symbol("a_y_0")]
    nx = df[trial_rows, Symbol("noise_x_0")]
    ny = df[trial_rows, Symbol("noise_y_0")]
    nvx = df[trial_rows, Symbol("noise_v_x_0")]
    nvy = df[trial_rows, Symbol("noise_v_y_0")]
    println("car pos: ", x_car[1], ", ", y_car[1], " car vel: ", vx_car[1], ", ", vy_car[1], " ped pos: ", x_ped[1], " , ", y_ped[1], " ped vel: ", vx_ped[1], " , ", vy_ped[1])
    ## TODO return agent trajectories

    create_actions(ax_ped, ay_ped, nx, ny, nvx, nvy)
end


# Load in the actions that caused a crash
println("loading actions")
file = "crashes_200.csv"
df = CSV.File(file) |> DataFrame
trial = 0
actions  = get_actions(df, trial)

println("Creating simulator and model")
# Setup the simulator to run it through
sim = AVSimulator()
car0 = Agent([-35,0], [11.17, 0])
peds0 = [Agent([0,-2], [0,1])]
model = PedestrianModel(0.1, 0.01, 0.1)

# Simulate
reward, car_traj, ped_traj = simulate(sim, actions, car0, peds0, model)
N = length(car_traj)
carx = [car_traj[i].pos[1] for i in 1:N]
cary = [car_traj[i].pos[2] for i in 1:N]

pedx = [ped_traj[i][1].pos[1] for i in 1:N]
pedy = [ped_traj[i][1].pos[2] for i in 1:N]
plot(carx, cary, marker=:square)
plot!(pedx, pedy, marker=:circ)

