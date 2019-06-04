using Random
using Statistics

include("train_ltl_auslan.jl")
include("train_gp_auslan.jl")

function kfolds(data, k, train_fn, model_loss_fn, model_callback = nothing)
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
            model_callback(model, loss, i)
        end
        push!(losses, loss)
    end
    return mean(losses), std(losses)
end

auslan_sign = "you" # Define a sign of interest

hand = rcols # Define the hand you want
vars = collect(keys(hand))

# Load in all examples for the sign of my choice
data = get_data(auslan_sign, vars, hand)

# m, s = kfolds(data, 5, train_ltl_model, ltl_model_loss, save_model)
# fname = "ltl_results.txt"

m, s = kfolds(data, 5, train_gp_model, gp_model_loss)
fname = "gp_results.txt"


f = open(fname, "w")
write(f, string("mean: ", m, " std: ", s))
close(f)

