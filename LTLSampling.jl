using Distributions
using DataStructures

# Action space struct that defines the bounds of a continuous action space
# As well as keeping a list of values that are not allowed
mutable struct ActionSpace
    # The bounds (min/max) of the action space. Entries like: :x => [xmin, xmax]
    bounds::OrderedDict{Symbol, Array{Float64,1}}
    # An entry like :x => [0,2] means x != 0 && x != 2
    not_equal::OrderedDict{Symbol, Array{Float64,1}}
end

# Get the action dimension of A
syms(A::ActionSpace) = collect(keys(A.bounds))
action_dim(A::ActionSpace) = length(syms(A))


Constraint = Pair{Expr, Bool}
Constraints = Array{Constraint}

# Construct an action space struct form a pair of symbols and min/max bounds
function ActionSpace(bounds::Pair...)
    bounds = OrderedDict(bounds...)
    not_equal = OrderedDict{Symbol, Array{Float64,1}}()
    for sym in keys(bounds)
        not_equal[sym] = Float64[]
    end
    ActionSpace(bounds, not_equal)
end

# check to see if this action space is valid
function action_space_valid(A::ActionSpace)
    for pair in A.bounds
        s, minb, maxb = pair[1], pair[2][1], pair[2][2]

        # Not valid if the max bound is less than the min bound. Or if the
        # bounds imply equality but the value is in the "not equals" constraints
        (maxb < minb || (minb == maxb && minb in A.not_equal[s])) && return false
    end
    true
end

# Induce the specified constraint on the minimum boundary
# If the new constraint is weaker then ignore it
function update_min!(A::ActionSpace, sym::Symbol, val)
    A.bounds[sym][1] = max(A.bounds[sym][1], val)
end

# Induce the specified constraint on the maximum boundary
# If the new constraint is weaker then ignore it.
function update_max!(A::ActionSpace, sym::Symbol, val)
    A.bounds[sym][2] = min(A.bounds[sym][2], val)
end

# Constrain the action space by the provided constraint
function constrain_action_space!(A::ActionSpace, constraint::Constraint)
    expr, truthval = constraint
    head, sym, val = (expr.head == :call) ? expr.args : [expr.head, expr.args...]
    if head == Symbol(".==")
        if truthval
            update_min!(A, sym, val)
            update_max!(A, sym, val)
        else
            push!(A.not_equal[sym], val)
        end
    elseif head == Symbol(".<")
        if truthval
            update_max!(A, sym, val)
        else
            update_min!(A, sym, val)
        end
    elseif head == Symbol(".>")
        if truthval
            update_min!(A, sym, val)
        else
            update_max!(A, sym, val)
        end
    else
        @error string("Unknown expression ", expr)
    end
end

# Apply the provided constraints to modify an action space
function constrain_action_space!(A::ActionSpace, constraints::Constraints)
    for constraint in constraints
        constrain_action_space!(A::ActionSpace, constraint)
    end
end

# Generate a new action space (deepcopy of input) and then constrain it
function constrain_action_space(A::ActionSpace, constraints::Constraints)
    Ac = deepcopy(A)
    constrain_action_space!(Ac, constraints)
    Ac
end

# Uniform sampling that handles equality and not equality constraints
function uniform_sample(vals, not_equal)
    # This is how equality is represented
    if vals[2] == vals[1]
        return vals[1]
    else
        # Here we sample a value and make sure it is not in our "not equality" list
        while true
            v = rand(Uniform(vals...))
            (v ∉ not_equal) && return v
        end
    end
end

# Get a sample form the provided action space
sample_action(A::ActionSpace, sym::Symbol) = uniform_sample(A.bounds[sym], A.not_equal[sym])


# Samples an expression that compares symbols to values using op
# Ch
function sample_sym_comparison(A::ActionSpace, op::Symbol)
    sym = rand(syms(A))
    val = sample_action(A, sym)
    Expr(:call, op, sym, val)
end


#######################################################################
#               Functions that govern inverse logic below             #
#######################################################################
terminals = [Symbol(".=="), Symbol(".<"), Symbol(".>")] # Calls that should terminate the tree search
expanders  = [:any, :all] # Calls the expand from scalar to time series
parameterized = [:all_before, :all_after]

all_before(τ, i) = all(τ[1:i])
all_after(τ, i) = all(τ[i:end])


function all_before_inv(out, i, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[1:i] .= true
    elseif out == false
        arr[rand(1:i)] .= false
    end
    (arr,)
end

function all_after_inv(out, i, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[i:N] .= true
    elseif out == false
        arr[rand(i:N)] .= false
    end
    (arr,)
end


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
        :all_before => all_before_inv,
        :all_after => all_after_inv,
        :.& => bitwise_and_inv,
        :.| => bitwise_or_inv
    )

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

# From a series of actions spaces, sample a series of actions
function sample_series(A_series)
    N, sym_list = length(A_series), syms(A_series[1])
    res = Dict(sym => Array{Float64}(undef, N) for sym in sym_list)
    for i=1:N
        for sym in sym_list
            res[sym][i] = sample_action(A_series[i], sym)
        end
    end
    res
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
    error("Could not find an sequence that satisifes the expression")
end


#########################################################################
#           Functions for managing trees (counting and pruning)         #
#########################################################################

# Count the number of nodes in an expression tree.
function count_nodes(tree, sum = 0)
    sum += 1
    for c in tree.children
        sum += count_nodes(c)
    end
    return sum
end

# Prune a tree until the cost function is above a certain value
function prune(tree, grammar, f, threshold, leaf_type, binary_terminals_to_ingore)
    pruned_tree  = tree
    while true
        leaves = get_leaves(pruned_tree, grammar, leaf_type)
        best_subtree, best_subtree_f = nothing, Inf
        for l in leaves
            new_tree = prune(pruned_tree, l, grammar, binary_terminals_to_ingore)
            new_f = f(new_tree, grammar)
            if new_f < best_subtree_f
                best_subtree, best_subtree_f = new_tree, new_f
            end
        end
        if best_subtree_f < threshold
            pruned_tree = best_subtree
        else
            break
        end
    end
    deepcopy(pruned_tree)
end

function get_leaves(tree, grammar, type = :τ)
    leaves = []
    if isempty(tree.children) && return_type(grammar, tree) == type
        push!(leaves, tree)
    end
    for c in tree.children
        push!(leaves, get_leaves(c, grammar, type)...)
    end
    leaves
end

function prune(tree::RuleNode, leaf, grammar, binary_terminals_to_ingore=:C)
    tree_cp = deepcopy(tree)
    prune!(tree_cp, leaf, grammar, binary_terminals_to_ingore)
    tree_cp
end

prune!(tree::RuleNode, leaf, grammar, binary_terminals_to_ingore=:C) = prune!([tree], leaf, grammar, binary_terminals_to_ingore)

function prune!(lineage::Array{RuleNode}, leaf, grammar, binary_terminals_to_ingore=:C)
    if lineage[end] == leaf
        # Move up the lineage until a binary operator is found
        for i in length(lineage)-1:-1:1
            ctypes = child_types(grammar, lineage[i].ind)
            if length(ctypes) == 2 && binary_terminals_to_ingore ∉ ctypes
                replace_tree!(lineage[i], i==1 ? nothing : lineage[i-1], lineage[i+1])
                return true
            end
        end
        return false
    end
    # Otherwise, true to prune the children
    for c in lineage[end].children
        prune!([lineage..., c], leaf, grammar, binary_terminals_to_ingore) && return true
    end
    false
end


# replace "tree" with one of its children (the one that is not child to remove)
# by replacing its spot in the parent.
# Assumes tree has exactly 2 children
function replace_tree!(tree, parent, child_to_remove)
    new_tree = filter(x->x!=child_to_remove, tree.children)[end]
    if isnothing(parent)
        tree.ind = new_tree.ind
        tree._val = new_tree._val
        while length(tree.children) > 0
            deleteat!(tree.children,1)
        end
        for c in new_tree.children
            push!(tree.children, c)
        end
    else
        tree_ind = findfirst(x->x==tree, parent.children)
        parent.children[tree_ind] = new_tree
    end
end

has_child(child, tree) = ~isempty(findall(x->x==child, tree.children))


