using Plots
pgfplots()

include("auslan_utils.jl")

d = get_data("you", collect(keys(rcols)), rcols)

Afull = ActionSpace(
    :thumb_bend => [0,1],
    :forefinger_bend => [0,1],
    :middlefinger_bend => [0,1],
    :ringfinger_bend => [0,1],
    :littlefinger_bend => [0,1],
    :roll => [-0.5, 0.5],
    :pitch => [-0.5, 0.5],
    :yaw => [-0.5, 0.5],
    :x_translation => [extrema(d[:x_translation][:])...],
    :y_translation => [extrema(d[:y_translation][:])...],
    :z_translation => [extrema(d[:z_translation][:])...]
    )
gp_dist = get_constrained_gp_dist(sek)

model = OrderedDict{Any,Any}(:x_translation=>:(all_between(x_translation .== 0.0143584, 10, 1) && all_after(x_translation .>= 0.0143584, 3)),:y_translation=>:(all_after(y_translation .== -0.136215, 10) && all_before(y_translation .>= x_translation, 8)),:z_translation=>:(all_after(z_translation .== -0.163841, 10)),:roll=>:(all(roll .>= 0.388889)),:pitch=>:(all_before(pitch .>= 0.388889, 10) && all_after(pitch .== 0.277778, 13)),:yaw=>:(all(yaw .>= 0.5)),:thumb_bend=>:(all_before(thumb_bend .== 0.888889, 6) && all_after(thumb_bend .== 0.111111, 11)),:forefinger_bend=>:(all(forefinger_bend .== 0.0)),:middlefinger_bend=>:(all_after(middlefinger_bend .<= 0.111111, 10) && all_before(middlefinger_bend .>= 1.0, 6)),:ringfinger_bend=>:(all_before(ringfinger_bend .== 0.888889, 6) && all_after(ringfinger_bend .<= 0.0, 10)),:littlefinger_bend=>:(all(littlefinger_bend .== thumb_bend)))

my_data = iterative_sample(Afull, [1.:15...], gp_dist, model)

p1 = plot(d[:y_translation], linecolor=:black, alpha=0.2, label="", title="Hand \$y\$-Position", ylabel="\$y\$-position")
plot!([14],[0.03],linecolor=:black, alpha=0, label="Training Samples")
plot!(my_data[:y_translation], linewidth=2, linecolor=:red, linestyle=:dash, label="Model Generated")


p2 = plot(d[:pitch], linecolor=:black, alpha=0.2, label="", title="Hand Pitch", ylabel="Pitch")
plot!([14],[0.3],linecolor=:black, alpha=0, label="Training Samples")
plot!(my_data[:pitch], linewidth=2, linecolor=:red, linestyle=:dash, label="Model Generated")

p3 = plot(d[:middlefinger_bend], linecolor=:black, alpha=0.2, label="", title="Middlefinger Bend", ylabel="Bend", xlabel="Timestep")
plot!([1],[1],linecolor=:black, alpha=0, label="Training Samples")
plot!(my_data[:middlefinger_bend], linewidth=2, linecolor=:red, linestyle=:dash, label="Model Generated")
plot(p1,p2,p3, layout=(3,1), size=(594,600))
savefig("test.pdf")





