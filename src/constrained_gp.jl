using LinearAlgebra

μ(X, m) = [m(x) for x in X]
K(X, X′, k) = [k(x,x′) for x in X, x′ in X′]

get_constrained_gp_dist(m, k) = (x, l, u, not_equal, n) -> sample_constrained_gp(m, k, x, l, u; n = n)

function sample_constrained_gp(m, k, Xν, l, u, σ2 = 1e-6; n = 1)
    N = length(l)
    @assert length(u) == N && length(not_equal) == N

    # Convert equality constraints to known points
    l,u = copy(l), copy(u)
    X, y = Xν[eq], l[eq]
    l[eq], u[eq] .= -Inf, Inf

    sample_constrained_gp(m, k, X, y, Xν, Xν, l, u, σ2; n = n)
end

# m - The mean function
# k - The kernel
# X - observation points
# y - observations
# Xs - prediction points
# Xv - virtual observation points
# σ2 - Variance of error in observations
# n - number of samples
function sample_constrained_gp(m, k, X, y, Xs, Xν, l, u, σ2 = 1e-6; n = 1)
    # Setup gram matrices
    N, Ns, Nν = length(X), length(Xs), length(Xν)
    Kxx = K(X, X, k)
    Kxxν = K(X, Xν, k)
    Kxxs = K(X, Xs, k)
    Kxνxν = K(Xν, Xν, k)
    Kxsxν = K(Xs, Xν, k)
    Kxsxs = K(Xs, Xs, k)
    Ixx = Matrix{Float64}(I, N, N)
    Ixsxs = Matrix{Float64}(I, Ns, Ns)
    Ixνxν = Matrix{Float64}(I, Nν, Nν)

    # Compute cholesky factors and important matrices
    L = cholesky(Kxx + σ2*Ixx).L
    v1 = L \ Kxxν
    v2 = L \ Kxxs
    A1 = (L' \ v1)'
    A2 = (L' \ v2)'
    B1 = Kxνxν + σ2*Ixsxs - v1' * v1
    B2 = Kxsxs - v2' * v2
    B3 = Kxsxν - v2' * v1

    L1 = cholesky(B1).L
    v3 = L1 \ B3'
    A = (L1' \ v3)'
    B = A2 - A * A1
    Σ = B2 - v3' * v3

    # Get mean vectors
    μo = μ(X, m)
    μs = μ(Xs, m)
    μν = μ(Xν, m)

    # Sample from the posterior
    Q = cholesky(Σ).L
    Cm = mvrandn_μ(μν + A1*(y - μo), l, u, B1, n)
    Um = rand(Normal(), (Ns, n))

    return μs + B*(y .- μo) .+ (A*(-μν .+ Cm) + Q*Um)
end







