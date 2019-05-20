using Distributions

function sample_until_neq(dist, not_equal)
    max_tries = 1000
    for i=1:max_tries
        v = rand(dist)
        (v ∉ not_equal) && return v
    end
    error("Couldn't find a sample that doesnt equal ", not_equal)
end

# Uniform sampling that handles equality and not equality constraints
function iid_samples(x, l, u, not_equal, n; dist = Uniform)
    N = length(x)
    @assert && length(l) == N && length(u) == N
    samps = zeros(N, n)
    for i=1:N
        if l[i] ≈ u[i]
            samps[i,:] .= l[i]
        else
            for j=1:n
                samps[i,j] = sample_until_neq(dist(l[i], u[i]), not_equal[i])
            end
        end
    end
end