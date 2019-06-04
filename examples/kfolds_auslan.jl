using Random
using Statistics

include("train_ltl_auslan.jl")
include("train_gp_auslan.jl")

function kfolds(data, k, train_fn, model_loss_fn, model_callback = nothing, label="")
    vars = collect(keys(hand))
    N = size(data[vars[1]], 2)
    test_size = convert(Int, floor(N/k))
    p = Random.randperm(N)
    losses = []
    for i in 1:k
        test_indices = p[test_size*(i-1) + 1 : test_size*i]
        train_indices = setdiff(p, test_indices)

        train_data = OrderedDict(a => data[a][:, train_indices] for a in keys(data))
        test_data = OrderedDict(a => data[a][:, test_indices] for a in keys(data))

        model, other_attr = train_fn(train_data)
        loss = model_loss_fn(test_data, model, other_attr)
        # loss = model_loss_fn(train_data, model, other_attr)
        if model_callback != nothing
            model_callback(model, loss, i, label)
        end
        push!(losses, loss)
    end
    return mean(losses), std(losses)
end

auslan_sign = "boy" # Define a sign of interest

hand = rcols # Define the hand you want
vars = collect(keys(hand))

# Load in all examples for the sign of my choice
data = get_data(auslan_sign, vars, hand)

m, s = kfolds(data, 5, train_ltl_model, ltl_model_loss, save_model, sign)
fname = string("ltl_results_", sign, ".txt")

# model = Dict(:x_translation => :(all_between((x_translation .== 0.014358444444444445) .& (x_translation .== 0.014358444444444445), 4, 7) && (any_between(x_translation .>= 0.014358444444444445, 10, 9) && all_between(x_translation .>= 0.014358444444444445, 1, 15))))
# using .LTLSampling
# include("auslan_utils.jl")
# Afull = ActionSpace(
#     :thumb_bend => [0,1],
#     :forefinger_bend => [0,1],
#     :middlefinger_bend => [0,1],
#     :ringfinger_bend => [0,1],
#     :littlefinger_bend => [0,1],
#     :roll => [-0.5, 0.5],
#     :pitch => [-0.5, 0.5],
#     :yaw => [-0.5, 0.5],
#     :x_translation => [-1,1],
#     :y_translation => [-1,1],
#     :z_translation => [-1,1]
#     )
# a = :y_translation
#
# ex = :(any_between(y_translation .== x_translation + 0.285456, 4, 13))
# x = [1.:15...]
# dist = get_constrained_gp_dist(sek)
# iterative_sample(a, ex, Afull, x, dist, model)

m, s = kfolds(data, 5, train_gp_model, gp_model_loss)
fname = "gp_results.txt"


f = open(fname, "w")
write(f, string("mean: ", m, " std: ", s))
close(f)

