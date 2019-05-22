
mutable struct TimeSeriesBounds
    l::Array{Float64,1}
    u::Array{Float64,1}
    neq::Array{Array{Float64,1}}
end

TimeSeriesBounds(N) = TimeSeriesBounds(zeros(N), zeros(N),  fill(Float64[], N))

# Get a list of expressions and outcomes that need to be satisfied for the top level expression to be true
function eval_conditional_tree(expr, desired_output, N)
    results = []
    eval_conditional_tree(expr, desired_output, results, N)
    results
end

# This version passes around a list of constraints (expression, values) pairs that need to be satisfied
# `expr` is an expression to sample a trajectory from
# `desired_output` - The desired output of the expression. Could be a scalar bool or a time series of bools
# `constraints` - The list of constraints and their corresponding truth values
# `N` - The length of the time series
function eval_conditional_tree(expr, desired_output, constraints, N)
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
        reduction = 0
        if op in expanders
            inv = inv(desired_output, N)
            eval_conditional_tree(expr.args[2], inv[1], constraints, N)
        elseif op in parameterized
            inv = inv(desired_output, expr.args[3], N)
            eval_conditional_tree(expr.args[2], inv[1], constraints, N)
            reduction = 1
        else
            inv = inv(desired_output)
            for i in 1:length(inv)
                eval_conditional_tree(expr.args[i+1], inv[i], constraints, N)
            end
        end

    else
        # Here "head" contains the operator and args contains the expressions
        # Get the inverse and recurse directly
        inv = bool_inverses[expr.head](desired_output)
        for i in 1:length(inv)
            eval_conditional_tree(expr.args[i], inv[i], constraints, N)
        end
    end
end

# Take the leaves from evalutating the conditional tree and turns them
# into a set of contraints for each timestep
function gen_constraints(expression_leaves, N)
    constraints = [Constraints(undef, 0) for i=1:N]
    for leaf in expression_leaves
        expr, truthvals = leaf
        for i in 1:N
            (truthvals[i] == :anybool) && continue
            push!(constraints[i], Constraint(expr, truthvals[i]))
        end
    end
    constraints
end

# Takes the constraints at each timestep and turns them into an action space
# at each timestep. Returns the action space at each timestep and bool indicating
# whether all of the timesteps have a valid action space
function gen_action_spaces(A, constraints)
    N = length(constraints)
    aspaces = Array{ActionSpace}(undef, N)
    valid_flag = Array{Bool}(undef, N)
    for i in 1:N
        aspaces[i] = constrain_action_space(A, constraints[i])
        valid_flag[i] = action_space_valid(A)
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
function sample_series(A_series, x, dist)
    @assert length(A_series) == length(x)
    lu_neq, sym_list, N = get_lu_neq(A_series), syms(A_series[1]), length(A_series)
    res = Dict(sym => Array{Float64}(undef, N) for sym in sym_list)

    for sym in sym_list
        res[sym] .= dist(x, lu_neq[sym].l,  lu_neq[sym].u, lu_neq[sym].neq)
    end
    res
end

# Sample a series of actions using the expression `ex` as a constraint on the
# actions space A.
# The algorithm will try `max_trials_for_valid` times before concluding that
# the constraints conflict with each other and no time series can be sampled
function sample_series(ex, A, x, dist; max_trials_for_valid = 10)
    N = length(x)
    leaves = eval_conditional_tree(ex, true, N)
    constraints = gen_constraints(leaves, N)
    for i in 1:max_trials_for_valid
        action_spaces, valid = gen_action_spaces(A, constraints)
        (valid) && return sample_series(action_spaces, x, dist)
    end
    error("Could not find an sequence that satisifes the expression")
end

