terminals = [Symbol(".=="), Symbol(".<="), Symbol(".>=")] # Calls that should terminate the tree search
expanders  = [:any, :all] # Calls the expand from scalar to time series
parameterized = Dict(:all_before => 1, :all_after => 1, :all_between => 2)

all_before(τ, i) = all(τ[1:i])
all_after(τ, i) = all(τ[i:end])

all_between(τ, i, j) = all(τ[min(i,j):max(i,j)])
any_between(τ, i, j) = any(τ[min(i,j):max(i,j)])

function all_between_inv(out, i, j, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
         arr[min(i,j):max(i,j)] .= true
    elseif out == false
        arr[rand(min(i,j):max(i,j))] = false
    end
    (arr,)
end

function any_between_inv(out, i, j, N)
    arr = Array{Any}(undef, N)
    fill!(arr, :anybool)
    if out == true
        pt = rand(min(i,j):max(i,j))
        arr[1:pt] .= false
        arr[pt] = true
    elseif out == false
        arr[min(i,j):max(i,j)] .= false
    end
    (arr,)
end


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
        :all_between => all_between_inv,
        :.& => bitwise_and_inv,
        :.| => bitwise_or_inv
    )