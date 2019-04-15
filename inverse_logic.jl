# This file contains the definitions of the inverse operations of different booleans.
# Given a desired output, these function returns possible inputs to the operator mentioned

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
    arr
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
    arr
end

# This function applies the inverse of a scalar boolean operation for each time
# `op_inv` is a function that specifies the scalar boolean operator inverse
# This function assumes that `op` takes in 2 arguments and outputs 1
function bitwise_op_inv(out, op_inv)
    N = length(out)
    arr1, arr2 = fill(false, N), fill(false, N)
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
