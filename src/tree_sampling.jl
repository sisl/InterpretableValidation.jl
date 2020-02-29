struct InvalidExpression
    msg::String
end

mutable struct TimeSeriesBounds
    l::Array{Float64,1}
    u::Array{Float64,1}
    neq::Array{Array{Float64,1}}
end

TimeSeriesBounds(N) = TimeSeriesBounds(zeros(N), zeros(N),  fill(Float64[], N))

# Get a list of expressions and outcomes that need to be satisfied for the top level expression to be true
function eval_conditional_tree(expr, desired_output, N)
    results = []
    eval_conditional_tree!(expr, desired_output, results, N)
    results
end

# This version passes around a list of constraints (expression, values) pairs that need to be satisfied
# `expr` is an expression to sample a trajectory from
# `desired_output` - The desired output of the expression. Could be a scalar bool or a time series of bools
# `constraints` - The list of constraints and their corresponding truth values
# `N` - The length of the time series
function eval_conditional_tree!(expr, desired_output, constraints, N)
    if expr.head == :call && expr.args[1] in terminals
        # Here we have hit a useful constraint expression.
        # Add it to the list of constraints and return
        push!(constraints, [expr, desired_output])
        return
    end
    if expr.head == :call
        # This could be :any, :all, :&, :|
        # Get the inverse of these operations and recurse
        # Special handing for expanding operators that need to be passed `N`
        op = expr.args[1]
        inv = bool_inverses[op]
        if op in expanders
            inv = inv(desired_output, N)
            eval_conditional_tree!(expr.args[2], inv[1], constraints, N)
        elseif op in keys(parameterized)
            ps = parameterized[op]
            inv = inv(desired_output, expr.args[3:3+(ps - 1)]..., N)
            eval_conditional_tree!(expr.args[2], inv[1], constraints, N)
        else
            inv = inv(desired_output)
            for i in 1:length(inv)
                eval_conditional_tree!(expr.args[i+1], inv[i], constraints, N)
            end
        end

    else
        # Here "head" contains the operator and args contains the expressions
        # Get the inverse and recurse directly
        inv = bool_inverses[expr.head](desired_output)
        for i in 1:length(inv)
            eval_conditional_tree!(expr.args[i], inv[i], constraints, N)
        end
    end
end

# Take the leaves from evalutating the conditional tree and turns them
# into a set of contraints for each timestep
function gen_constraints(expression_leaves, N, sym_data = Dict())
    passed_syms = collect(keys(sym_data))
    constraints = [Constraints(undef, 0) for i=1:N]
    for leaf in expression_leaves
        expr, truthvals = leaf
        rsyms = passed_syms[[occursin(string(s), string(expr)) for s in passed_syms]]
        for i in 1:N
            (truthvals[i] == :anybool) && continue
            if !isempty(rsyms)
                new_expr = string(expr)
                for rsym in rsyms
                    new_expr = replace(new_expr, string(rsym) => string(sym_data[rsym][i]))
                end
                push!(constraints[i], Constraint(Meta.parse(new_expr), truthvals[i]))
            else
                push!(constraints[i], Constraint(expr, truthvals[i]))
            end
        end
    end
    constraints
end

# Takes the constraints at each timestep and turns them into an action space
# at each timestep. Returns the action space at each timestep and bool indicating
# whether all of the timesteps have a valid action space
function gen_action_spaces(A::Atype, constraints) where Atype
    N = length(constraints)
    aspaces = Array{Atype}(undef, N)
    valid_flag = Array{Bool}(undef, N)
    for i in 1:N
        aspaces[i] = constrain_action_space(A, constraints[i])
        valid_flag[i] = action_space_valid(aspaces[i])
    end
    aspaces, all(valid_flag)
end

# Convert a series of action spaces to an array of lower/upper bounds and neq constraints
function get_lu_neq(A_series)
    N, sym_list = length(A_series), syms(A_series[1])
    res = Dict(sym => TimeSeriesBounds(N) for sym in sym_list)
    for i=1:N
        for sym in sym_list
            res[sym].l[i] = A_series[i].bounds[sym][1]
            res[sym].u[i] = A_series[i].bounds[sym][2]
            res[sym].neq[i] = A_series[i].not_equal[sym]
        end
    end
    res
end

# From a series of actions spaces, sample a time series
# dist - sampling distribution that can handle constraints
# x - locations at which to sample (important if there are temporal correlations)
# n - Number of samples per action
function sample_series(A_series::Array{ActionSpace}, x, dist)
    @assert length(A_series) == length(x)
    lu_neq, sym_list, N = get_lu_neq(A_series), syms(A_series[1]), length(A_series)
    res = Dict(sym => Array{Float64}(undef, N) for sym in sym_list)

    for sym in sym_list
        res[sym] .= dist(x, lu_neq[sym].l,  lu_neq[sym].u, lu_neq[sym].neq)
    end
    res
end

function sample_discrete(A::DiscreteActionSpace, sym, dist::Dict{Int, Float64})
    probs = zeros(length(dist))
    for i in A.feasible[sym]
        probs[i] = dist[i]
    end
    probs ./= sum(probs)
    return rand(Categorical(probs))
end


function sample_series(A_series::Array{DiscreteActionSpace}, x, dist::Dict{Int, Float64})
    @assert length(A_series) == length(x)
    sym_list, N = syms(A_series[1]), length(A_series)
    res = OrderedDict(sym => Array{Int64}(undef, N) for sym in sym_list)

    for sym in sym_list
        for i=1:N
            res[sym][i] = sample_discrete(A_series[i], sym, dist)
        end
    end
    res
end

# Sample a series of actions using the expression `ex` as a constraint on the
# actions space A.
# The algorithm will try `max_trials_for_valid` times before concluding that
# the constraints conflict with each other and no time series can be sampled
function sample_series(ex, A, x, dist, sym_data = Dict(); max_trials_for_valid = 10)
    N = length(x)
    leaves = eval_conditional_tree(ex, true, N)
    constraints = gen_constraints(leaves, N, sym_data)
    for i in 1:max_trials_for_valid
        action_spaces, valid = gen_action_spaces(A, constraints)
        (valid) && return sample_series(action_spaces, x, dist)
    end
    throw(InvalidExpression("Expression was invalid"))
end

function iterative_sample(Afull, x, dist, model)
    # Get a sample of all actions up to this point
    data = Dict()
    for a in keys(model)
        data[a] = sample_series(model[a], ActionSpace(Afull, a), x, dist, data)[a]
    end
    return data
end

# This function samples trajectories in an iterative manner looping over each dimension.
# this way we can describe relationships between variables and we build them up one at a time
function iterative_sample(a, ex, Afull, x, dist, found_expressions)
    curr_data = iterative_sample(Afull, x, dist, found_expressions)
    # Get a sample of the current expression
    curr_data[a] = sample_series(ex, ActionSpace(Afull, a), x, dist, curr_data)[a]
    return curr_data
end

