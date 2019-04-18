# This file contains the definitions of the inverse operations of different booleans.
# Given a desired output, these function returns possible inputs to the operator mentioned

include("sampling.jl")

# Inverse of the "and" operator which takes two boolean inputs
# If output is true then both inputs are true
# If the output is false then at least one of the inputs is false
function and_inv(out)
    if out == :anybool
        return (:anybool, :anybool)
    elseif out
        return (true, true)
    else
        return rand([(false, :anybool), (:anybool, false)])
    end
end

# Inverse of the "or" operator which takes two boolean inputs
# If the output is true then at least one of the inputs is true
# If the output is false then both if the inputs are false
function or_inv(out)
    if out == :anybool
        return (:anybool, :anybool)
    elseif out
        return rand([(true, :anybool), (:anybool, true)])
    else
        return (false, false)
    end
end

# Inverse of the "all" or "globally" operator which takes a vector of bools
# If the output is true then all inputs in the time series are true
# If the output is false then there is at least on false in the time series
function all_inv(out, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         fill!(arr, true)
    elseif out == false
        arr[rand(1:N)] = false
    end
    (arr,)
end

# Inverse of the "any" or "eventually" operator which takes a vector of bools
# If the output is true then there is at least on true in the time series
# If the output is false then al
function any_inv(out, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
        arr[rand(1:N)] = true
    elseif out == false
        fill!(arr, false)
    end
    (arr,)
end

# This function applies the inverse of a scalar boolean operation for each time
# `op_inv` is a function that specifies the scalar boolean operator inverse
# This function assumes that `op` takes in 2 arguments and outputs 1
function bitwise_op_inv(out, op_inv)
    N = length(out)
    arr1, arr2 = Array{Any}(undef, N), Array{Any}(undef, N)
    for i in 1:N
        arr1[i], arr2[i] = op_inv(out[i])
    end
    return (arr1, arr2)
end

# Defines the invers of the bitwise "and" operator
bitwise_and_inv(out) = bitwise_op_inv(out, and_inv)

# Defines the inverse of the bitwise "or" operator
bitwise_or_inv(out) = bitwise_op_inv(out, or_inv)

bool_inverses = Dict(
        :&& => and_inv,
        :|| => or_inv,
        :any => any_inv,
        :all => all_inv,
        :.& => bitwise_and_inv,
        :.| => bitwise_or_inv
    )

terminals = [Symbol("=="), Symbol("<"), Symbol(">")] # Calls that should terminate the tree search
expanders  = [:any, :all] # Calls the expand from scalar to time series

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
        inv = (op in expanders) ? inv(desired_output, N) : inv(desired_output)
        for i in 1:length(inv)
            eval_conditional_tree(expr.args[i+1], inv[i], constraints, N)
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
    return constraints
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
    return aspaces, all(valid_flag)
end

# From a series of actions spaces, sample a series of actions
function sample_series(A_series)
    N = length(A_series)
    ndim = action_dim(A_series[1])
    res = Array{Float64}(undef, N, ndim)
    for i=1:N
        res[i, :] = sample_action(A_series[i])
    end
    return res
end


# Sample a series of actions using the expression `ex` as a constraint on the
# actions space A.
# The algorithm will try `max_trials_for_valid` times before concluding that
# the constraints conflict with each other and no time series can be sampled
function sample_series(ex, A, N, max_trials_for_valid = 10)
    leaves = eval_conditional_tree(ex, true, N)
    constraints = gen_constraints(leaves, N)
    for i in 1:max_trials_for_valid
        action_spaces, valid = gen_action_spaces(A, constraints)
        (valid) && return sample_series(action_spaces)
    end
    @error "Could not find an sequence that satisifes the expression"
end

