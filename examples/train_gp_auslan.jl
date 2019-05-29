using GaussianProcesses
using Plots
using LinearAlgebra
function gp_model_loss(test_data, model, other_attr)
    vars = collect(keys(test_data))
    T = size(test_data[vars[1]], 1)
    total_loss = 0
    for a in vars
        println("a: ", a)
        actions = []
        try
            actions = predict_y(model[a],[1.:T...])[1]
        catch e
            if isa(e, PosDefException)
                println("Positive definite exception. returning 0s")
                actions = zeros(T)
            else
                error("Uncaught exception, ", e)
            end
        end
        total_loss += time_series_loss(test_data[a], actions)
    end
    return total_loss
end

function train_gp_model(train_data)
    vars = collect(keys(train_data))
    T = size(train_data[vars[1]], 1)
    x = repeat([1.:T...], size(train_data[vars[1]], 2))

    model = OrderedDict()
    for a in keys(train_data)
        println("Training gp for ", a)
        y = train_data[a][:]
        gp = GP(x, y, MeanZero(), SE(0.0, 0.0))
        optimize!(gp)
        model[a] = gp
    end
    return model, nothing
end

