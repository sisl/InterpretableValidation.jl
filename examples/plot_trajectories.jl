using Plots
pyplot()
using AutoViz
using AutomotiveDrivingModels
using Cairo
# include("av_simulator.jl")
# include("setup_av_scenario.jl")
# using .LTLSampling

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
        plot_traj(car_traj, ped_traj, p)
    end
    p
end

function plot_traj(car_traj, ped_traj, p=nothing)
    if p == nothing
        p = plot(legend = :topleft, size=(700,300), xlims=(-19, 3))
    end
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


struct Intersection <: SceneOverlay end

function AutoViz.render!(rendermodel::RenderModel, underlay::Intersection, scene, roadway)

    h0 = -3.5
    h = 1.
    s = 0.8

    add_instruction!(rendermodel, render_rect, (-3, h0, 6, h, colorant"yellow", true, true, colorant"grey"))
    add_instruction!(rendermodel, render_rect, (-3, h0 + (h+s), 6, h, colorant"yellow", true, true, colorant"grey"))
    add_instruction!(rendermodel, render_rect, (-3, h0 + 2*(h+s), 6, h, colorant"yellow", true, true, colorant"grey"))
    add_instruction!(rendermodel, render_rect, (-3, h0 + 3*(h+s), 6, h, colorant"yellow", true, true, colorant"grey"))
    add_instruction!(rendermodel, render_rect, (-3, h0 + 4*(h+s), 6, h, colorant"yellow", true, true, colorant"grey"))
    add_instruction!(rendermodel, render_rect, (-3, h0 + 5*(h+s), 6, h, colorant"yellow", true, true, colorant"grey"))

    return rendermodel
end


function myrender(scene::EntityFrame{S,D,I}, roadway::R, overlays::AbstractVector, underlays::AbstractVector;
    canvas_width::Int=DEFAULT_CANVAS_WIDTH,
    canvas_height::Int=DEFAULT_CANVAS_HEIGHT,
    rendermodel::RenderModel=RenderModel(),
    cam::Camera=SceneFollowCamera(),
    car_colors::Dict{I,C}=Dict{I,Colorant}(),
    surface::CairoSurface = CairoSVGSurface(IOBuffer(), canvas_width, canvas_height)
    ) where {S,D,I,R,C<:Colorant}


    ctx = creategc(surface)
    clear_setup!(rendermodel)


    render!(rendermodel, roadway)
    for underlay in underlays
        render!(rendermodel, underlay, scene, roadway)
    end
    render!(rendermodel, scene, car_colors=car_colors)

    for overlay in overlays
        render!(rendermodel, overlay, scene, roadway)
    end

    camera_set!(rendermodel, cam, scene, roadway, canvas_width, canvas_height)

    render(rendermodel, ctx, canvas_width, canvas_height)
    return surface
end


function autoviz_traj(veh, ped, ped_noise)
    LP = VecE2(-50,0)
    C = VecE2(25,0)
    dy = VecE2(0, DEFAULT_LANE_WIDTH / 2)


    roadway = Roadway()

    curve = gen_straight_curve(LP, C, 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    curve = gen_straight_curve(LP + 2*dy, C + 2*dy, 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    scene= Scene()
    push!(scene, Vehicle(VehicleState(VecSE2(veh.pos...), norm(veh.vel)), VehicleDef(), 1))
    push!(scene, Vehicle(VehicleState(VecSE2(ped.pos..., atan(reverse(ped.vel)...)), norm(ped.vel)), VehicleDef(AgentClass.PEDESTRIAN, 1.5,1.5), 2))
    push!(scene, Vehicle(VehicleState(VecSE2(ped_noise.pos..., atan(reverse(ped_noise.vel)...)), norm(ped_noise.vel)), VehicleDef(AgentClass.PEDESTRIAN, 1.5,1.5), 3))


    cam = FitToContentCamera(.1)
    car_colors = Dict(1 => colorant"blue", 2=>colorant"red", 3=>colorant"gray")
    underlays = [ Intersection()]

    myrender(scene, roadway, [], underlays, cam=cam, car_colors = car_colors)
end

autoviz_traj(veh[1], ped[1][1])

ped[1][1].vel
atan(ped[1][1].vel[2], ped[1][1].vel[1])

# ex = Meta.parse("all(((ax .<= 0.1291635133369189) .& (ny .== -0.9742963087069332)) .& (ay .== -0.30090443707542924))")
#
# plot_examples(ex, 1, "")
# savefig("ltl_av_example.pdf")

