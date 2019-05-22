using Distributions

function sample_until_neq(dist, not_equal)
    max_tries = 1000
    for i=1:max_tries
        v = rand(dist)
        (v ∉ not_equal) && return v
    end
    error("Couldn't find a sample that doesnt equal ", not_equal)
end

uniform_sample(l, u, not_equal) = iid_samples([1], [l], [u], [not_equal])[1]

# Uniform sampling that handles equality and not equality constraints
function iid_samples(x, l, u, not_equal; dist = Uniform)
    N = length(x)
    @assert length(l) == N && length(u) == N
    samps = zeros(N)
    for i=1:N
        if l[i] ≈ u[i]
            samps[i] = l[i]
        else
            samps[i] = sample_until_neq(dist(l[i], u[i]), not_equal[i])
        end
    end
    samps
end

