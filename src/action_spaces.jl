using DataStructures

# Action space struct that defines the bounds of a continuous action space
# As well as keeping a list of values that are not allowed
mutable struct ActionSpace
    # The bounds (min/max) of the action space. Entries like: :x => [xmin, xmax]
    bounds::OrderedDict{Symbol, Array{Float64,1}}
    # An entry like :x => [0,2] means x != 0 && x != 2
    not_equal::OrderedDict{Symbol, Array{Float64,1}}
end

# Construct an action space struct form a pair of symbols and min/max bounds
function ActionSpace(bounds::Pair...)
    bounds = OrderedDict(bounds...)
    not_equal = OrderedDict{Symbol, Array{Float64,1}}()
    for sym in keys(bounds)
        not_equal[sym] = Float64[]
    end
    ActionSpace(bounds, not_equal)
end

# Construct an actions space from just one symbol in another action space
ActionSpace(A::ActionSpace, a::Symbol) = ActionSpace(OrderedDict(a => A.bounds[a]), OrderedDict(a => A.not_equal[a]))

Constraint = Pair{Expr, Bool}
Constraints = Array{Constraint}


# Get the action dimension of A
syms(A::ActionSpace) = collect(keys(A.bounds))
action_dim(A::ActionSpace) = length(syms(A))



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
function update_min!(A::ActionSpace, sym::Symbol, val::Float64)
    A.bounds[sym][1] = max(A.bounds[sym][1], val)
end

update_min!(A::ActionSpace, sym::Symbol, val::Expr) = update_min!(A, sym, eval(val))

# Induce the specified constraint on the maximum boundary
# If the new constraint is weaker then ignore it.
function update_max!(A::ActionSpace, sym::Symbol, val::Float64)
    A.bounds[sym][2] = min(A.bounds[sym][2], val)
end

update_max!(A::ActionSpace, sym::Symbol, val::Expr) = update_max!(A, sym, eval(val))

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
    elseif head == Symbol(".<=")
        if truthval
            update_max!(A, sym, val)
        else
            update_min!(A, sym, val)
        end
    elseif head == Symbol(".>=")
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

# Get a sample from the provided action space
sample_uniform_action(A::ActionSpace, sym::Symbol) = uniform_sample(A.bounds[sym]..., A.not_equal[sym])
sample_uniform_action(A::ActionSpace) = [sample_uniform_action(A, sym) for sym in syms(A)]

# Samples an expression that compares symbols to values using op
function sample_sym_comparison(A::ActionSpace, op::Symbol)
    sym = rand(syms(A))
    val = sample_uniform_action(A, sym)
    Expr(:call, op, sym, val)
end
