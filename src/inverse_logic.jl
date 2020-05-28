const IV_TERMINALS = [Symbol(".=="), Symbol(".<="), Symbol(".>=")] # Calls that should terminate the tree search
const IV_EXPANDERS  = [:any, :all] # Calls the expand from scalar to time series
const IV_PARAMETERIZED = Dict(:all_before => 1, :all_after => 1, :all_between => 2, :any_between=>2)

# Negation operator applied to bool or :anybool
flex_not(b::Union{Symbol, Bool}) =  (b == :anybool ? :anybool : !b)

all_before(τ, i) = all(τ[1:i])
all_after(τ, i) = all(τ[i:end])

all_between(τ, i, j) = all(τ[min(i,j):max(i,j)])
any_between(τ, i, j) = any(τ[min(i,j):max(i,j)])

function all_between_inv(out, i, j, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[min(i,j):max(i,j)] .= true
    elseif out == false
        arr[rand(rng, min(i,j):max(i,j))] = false
    end
    (arr,)
end

function any_between_inv(out, i, j, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
        pt = rand(rng, min(i,j):max(i,j))
        arr[1:pt] .= false
        arr[pt] = true
    elseif out == false
        arr[min(i,j):max(i,j)] .= false
    end
    (arr,)
end


function all_before_inv(out, i, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[1:i] .= true
    elseif out == false
        arr[rand(rng, 1:i)] = false
    end
    (arr,)
end

function all_after_inv(out, i, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[i:N] .= true
    elseif out == false
        arr[rand(rng, i:N)] = false
    end
    (arr,)
end


# Inverse of the "and" operator which takes two boolean inputs
# If output is true then both inputs are true
# If the output is false then at least one of the inputs is false
function and_inv(out, rng::AbstractRNG)
    if out == :anybool
        return (:anybool, :anybool)
    elseif out
        return (true, true)
    else
        return rand(rng, [(false, :anybool), (:anybool, false)])
    end
end

# Inverse of the "or" operator which takes two boolean inputs
# If the output is true then at least one of the inputs is true
# If the output is false then both if the inputs are false
function or_inv(out, rng::AbstractRNG)
    if out == :anybool
        return (:anybool, :anybool)
    elseif out
        return rand(rng, [(true, :anybool), (:anybool, true)])
    else
        return (false, false)
    end
end

# Inverse of the "all" or "globally" operator which takes a vector of bools
# If the output is true then all inputs in the time series are true
# If the output is false then there is at least on false in the time series
function all_inv(out, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         fill!(arr, true)
    elseif out == false
        arr[rand(rng, 1:N)] = false
    end
    (arr,)
end

# Inverse of the "any" or "eventually" operator which takes a vector of bools
# If the output is true then there is at least on true in the time series
# If the output is false then al
function any_inv(out, N, rng::AbstractRNG)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
        arr[rand(rng, 1:N)] = true
    elseif out == false
        fill!(arr, false)
    end
    (arr,)
end

# This function applies the inverse of a scalar boolean operation for each time
# `op_inv` is a function that specifies the scalar boolean operator inverse
# This function assumes that `op` takes in 2 arguments and outputs 1
function bitwise_op_inv(out, op_inv, rng::AbstractRNG)
    N = length(out)
    arr1, arr2 = Array{Any}(undef, N), Array{Any}(undef, N)
    for i in 1:N
        arr1[i], arr2[i] = op_inv(out[i], rng)
    end
    return (arr1, arr2)
end

# Defines the invers of the bitwise "and" operator
bitwise_and_inv(out, rng::AbstractRNG) = bitwise_op_inv(out, and_inv, rng)

# Defines the inverse of the bitwise "or" operator
bitwise_or_inv(out, rng::AbstractRNG) = bitwise_op_inv(out, or_inv, rng)

# Mapping an operation to its inverse
bool_inverses = Dict(
        :&& => and_inv,
        :|| => or_inv,
        :any => any_inv,
        :all => all_inv,
        :all_before => all_before_inv,
        :all_after => all_after_inv,
        :all_between => all_between_inv,
        :any_between => any_between_inv,
        :.& => bitwise_and_inv,
        :.| => bitwise_or_inv
    )

# Get a list of expressions and outcomes that need to be satisfied for the top level expression to be true
function sample_constraints(expr, N, rng::AbstractRNG)
    constraints = []
    sample_constraints!(expr, true, constraints, N, rng)
    constraints
end

# This version passes around a list of constraints (expression, values) pairs that need to be satisfied
# `expr` is an expression to sample a trajectory from
# `truthval` - The desired output of the expression. Could be a scalar bool or a time series of bools
# `constraints` - The list of constraints and their corresponding truth values
# `N` - The length of the time series
function sample_constraints!(expr, truthval, constraints, N, rng::AbstractRNG)
    if expr.head == :call && expr.args[1] in IV_TERMINALS
        # Add constrain expression to the list of constraints and return
        push!(constraints, [expr, truthval])
        return
    end
    if expr.head == :call
        # This could be :any, :all, :&, :|
        # Get the inverse of these operations and recurse
        # Special handing for expanding operators that need to be passed `N`
        op = expr.args[1]
        inv = bool_inverses[op]
        if op in IV_EXPANDERS
            inv = inv(truthval, N, rng)
            sample_constraints!(expr.args[2], inv[1], constraints, N, rng)
        elseif op in keys(IV_PARAMETERIZED)
            ps = IV_PARAMETERIZED[op]
            inv = inv(truthval, expr.args[3:3+(ps - 1)]..., N, rng)
            sample_constraints!(expr.args[2], inv[1], constraints, N, rng)
        else
            inv = inv(truthval, rng)
            for i in 1:length(inv)
                sample_constraints!(expr.args[i+1], inv[i], constraints, N, rng)
            end
        end

    else
        # Here "head" contains the operator and args contains the expressions
        # Get the inverse and recurse directly
        inv = bool_inverses[expr.head](truthval, rng)
        for i in 1:length(inv)
            sample_constraints!(expr.args[i], inv[i], constraints, N, rng)
        end
    end
end

