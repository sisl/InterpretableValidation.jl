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
function action_space_valid(A)
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
function update_min!(A, sym, val)
    A.bounds[sym][1] = max(A.bounds[sym][1], val)
end

# Induce the specified constraint on the maximum boundary
# If the new constraint is weaker then ignore it.
function update_max!(A, sym, val)
    A.bounds[sym][2] = min(A.bounds[sym][2], val)
end

# Constrain the action space by the provided constraint
function constrain_action_space!(A, constraint::Constraint)
    expr, truthval = constraint
    head, sym, val = (expr.head == :call) ? expr.args : [expr.head, expr.args...]
    if head == Symbol("==")
        if truthval
            update_min!(A, sym, val)
            update_max!(A, sym, val)
        else
            push!(A.not_equal[sym], val)
        end
    elseif head == :<
        if truthval
            update_max!(A, sym, val)
        else
            update_min!(A, sym, val)
        end
    elseif head == :>
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
function constrain_action_space!(A, constraints::Constraints)
    for constraint in constraints
        constrain_action_space!(A, constraint)
    end
end

# Generate a new action space (deepcopy of input) and then constrain it
function constrain_action_space(A, constraints::Constraints)
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
sample_action(A) = [uniform_sample(p[2], A.not_equal[p[1]]) for p in A.bounds]



