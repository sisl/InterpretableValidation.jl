# using .LTLSampling
using Statistics
using ExprRules
using ExprOptimization
include("auslan_utils.jl")

function save_model(model, loss, i)
    file = string("ltl_model_", i, ".txt")
    f = open(file, "w")
    write(f, string(model, "\n"))
    write(f, string(loss))
    close(f)
end

function ltl_model_loss(test_data, model, other_attr)
    Afull = other_attr["Afull"]
    x = other_attr["x"]
    gp_dist = other_attr["dist"]

    total_loss, trials = 0, 10

    for i=1:trials
        actions = []
        try
            actions = iterative_sample(Afull, x, gp_dist, model)
        catch e
            if isa(e, InvalidExpression)
                return 1e9
            else
                println("uncaught error, ", e)
                error("Uncaught error, ", e)
            end
        end
        akeys = collect(keys(actions))
        for a in akeys
            total_loss += time_series_loss(test_data[a], actions[a])
        end
    end
    total_loss/trials
end

function train_ltl_model(train_data)
    vars = collect(keys(train_data))
    T = size(train_data[vars[1]], 1)
    xx = [1.:T...]
    indices = [1:T...]

    # Express the full action space
    Afull = ActionSpace(
        :thumb_bend => [0,1],
        :forefinger_bend => [0,1],
        :middlefinger_bend => [0,1],
        :ringfinger_bend => [0,1],
        :littlefinger_bend => [0,1],
        :roll => [-0.5, 0.5],
        :pitch => [-0.5, 0.5],
        :yaw => [-0.5, 0.5],
        :x_translation => [extrema(train_data[:x_translation][:])...],
        :y_translation => [extrema(train_data[:y_translation][:])...],
        :z_translation => [extrema(train_data[:z_translation][:])...]
        )

    # Build a gaussian process with the correct kernel
    gp_dist = get_constrained_gp_dist(sek)

    # Define a dictionary of expressions for each dimension
    model = OrderedDict()
    grammar = nothing

    # Loop through the action dimensions and solve the optimization problem one at a time
    for a in vars
        println("Optimizing expression for ", a)
        known_vals = collect(keys(model))

        # Define an action space for  the single action
        A = ActionSpace(a => Afull.bounds[a])
        rng = range(A.bounds[a]..., length=10)

        # Define a grammar
        grammar = Meta.eval(macroexpand(Main, Meta.parse(string("@grammar begin
            R = (R && R)
            R = all(τ) | any(τ) | all_before(τ, C) | all_after(τ, C) | all_between(τ, C, C)
            C = |($indices)
            τ = (τ .& τ)
            τ = ($a .<= G) | ($a .== G) | ($a .>= G)
            G = |($rng)", isempty(known_vals) ? "" : "
            G = H
            H = |($known_vals)
            H = (H + G) | (H*G) | (H - G)","
        end"))))

        # Define a loss
        function loss(rn::RuleNode, grammar::Grammar)
            ex = get_executable(rn, grammar)
            total_loss, trials = 0, 10

            for i=1:trials
                actions = []
                try
                    actions = iterative_sample(a, ex, Afull, xx, gp_dist, model)[a]
                catch e
                    if isa(e, InvalidExpression)
                        return 1e9
                    else
                        println("uncaught error, ", e, " on expr: ", ex)
                        error("Uncaught error, ", e)
                    end
                end
                total_loss += time_series_loss(train_data[a], actions)
            end

            total_loss/trials
        end

        # Optimize the function
        p = GeneticProgram(1000,30,6,0.3,0.3,0.4)
        results_gp = optimize(p, grammar, :R, loss, verbose = true)
        tree = prune_unused_nodes(results_gp.tree, grammar, loss, 15*results_gp.loss, :τ, [:C, :G, :H])

        # store the result
        model[a] = get_executable(tree, grammar)

        println("found expression: ", results_gp.expr, " for ", a, " with loss: ", results_gp.loss)
    end
    other_attr = Dict("Afull" => Afull, "x" => xx, "dist" => gp_dist)
    model, other_attr
end

