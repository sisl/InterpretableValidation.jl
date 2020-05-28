# This is the baseline distribution (i.e. a guassian process or iid uniform)
abstract type TimeseriesDistribution end

## Time series that are independent and identically distributed
struct IID{D} <: TimeseriesDistribution where {D}
    x::Array{Float64} # Sample locations
    distribution::D # Sampling distributions
end

# Number of points in the time series
N_pts(t::TimeseriesDistribution) = length(t.x)

# logpdf is just the summation over the individual values
Distributions.logpdf(iid::IID{D}, y) where {D} = sum([logpdf(iid.distribution, yi) for yi in y])

# Uniform distribution
function Base.rand(rng::AbstractRNG, iid::IID{Uniform{Float64}}, lb::Array{Float64} = -Inf*ones(N_pts(iid)), ub::Array{Float64} = Inf*ones(N_pts(iid)))
    samples = Array{Float64}(undef, length(iid.x))
    l, u = max.(lb, iid.distribution.a),  min.(ub, iid.distribution.b)
    for i=1:length(iid.x)
        samples[i] = rand(rng, Uniform(l[i], u[i]))
    end
    samples
end

# Gaussian Distribution
function Base.rand(rng::AbstractRNG, iid::IID{Normal{Float64}}, lb::Array{Float64} = -Inf*ones(N_pts(iid)), ub::Array{Float64} = Inf*ones(N_pts(iid)))
    samples = Array{Float64}(undef, length(iid.x))
    for i=1:length(iid.x)
        samples[i] = rand(rng, Truncated(iid.distribution, lb[i], ub[i]))
    end
    samples
end

# Categorical
function Base.rand(rng::AbstractRNG, iid::IID{Categorical{Float64, Array{Float64,1}}}, feasible::Array{Array{Int64,1},1} = fill(collect(1:length(iid.distribution.p)), N_pts(iid)))
    samples = Array{Int64}(undef, length(iid.x))
    for i=1:length(iid.x)
        probs = zeros(length(iid.distribution.p))
        probs[feasible[i]] = iid.distribution.p[feasible[i]]
        d = Categorical(probs ./ sum(probs))
        samples[i] = rand(rng, d)
    end
    samples
end

## Time series distributed according to a Gaussian Processs
squared_exp_kernel(;l=2, σ2 = 1) = (x,xp) -> σ2*exp(-(x-xp)^2/(2*l^2))

# Construct the mean vector
μ(X::Array{Float64}, m) = [m(x) for x in X]

# Construct the covariance matrix
K(X::Array{Float64}, X′::Array{Float64}, k) = [k(x,x′) for x in X, x′ in X′]

# Struct defining a Guassian Process
@with_kw struct GaussianProcess <: TimeseriesDistribution
    m::Function # Mean function (x_i) -> μ_i
    k::Function # Kernel function (x_i, x_j) -> k_ij
    x::Array{Float64} # Sample locations
    σ2::Float64 = 1e-6 # std deviation of noise on the samples
end

# Generate a random sample from the gaussian process
function Base.rand(rng::AbstractRNG, gp::GaussianProcess, lb::Array{Float64} = -Inf*ones(N_pts(gp)), ub::Array{Float64} = Inf*ones(N_pts(gp)))
    Xν, Xs, X, y = gp.x, gp.x, Float64[], Float64[]

    # Convert equality constraints to known points
    eq = lb .≈ ub
    l, u = copy(lb), copy(ub)
    X, y = Float64[X..., Xν[eq]...], Float64[y..., l[eq]...]
    l[eq] .= -Inf
    u[eq] .= Inf

    # Setup gram matrices
    N, Ns, Nν = length(X), length(Xs), length(Xν)
    Kxx = K(X, X, gp.k)
    Kxxν = K(X, Xν, gp.k)
    Kxxs = K(X, Xs, gp.k)
    Kxνxν = K(Xν, Xν, gp.k)
    Kxsxν = K(Xs, Xν, gp.k)
    Kxsxs = K(Xs, Xs, gp.k)
    Ixx = Matrix{Float64}(I, N, N)
    Ixsxs = Matrix{Float64}(I, Ns, Ns)
    Ixνxν = Matrix{Float64}(I, Nν, Nν)

    # Compute cholesky factors and important matrices
    L = cholesky(Kxx + gp.σ2*Ixx).L
    v1 = L \ Kxxν
    v2 = L \ Kxxs
    A1 = (L' \ v1)'
    A2 = (L' \ v2)'
    B1 = Kxνxν + gp.σ2*Ixνxν - v1' * v1
    B2 = Kxsxs + gp.σ2*Ixsxs - v2' * v2
    B3 = Kxsxν - v2' * v1

    L1 = cholesky(B1).L
    v3 = L1 \ B3'
    A = (L1' \ v3)'
    B = A2 - A * A1
    Σ = B2 - v3' * v3

    # Get mean vectors
    μo = μ(X, gp.m)
    μs = μ(Xs, gp.m)
    μν = μ(Xν, gp.m)

    # Sample from the posterior
    Q = cholesky(Σ).L
    Cm = mvrandn_μ(rng, μν + A1*(y - μo), l, u, B1, 1)
    Um = rand(rng, Normal(), (Ns, 1))

    dropdims(μs + B*(y .- μo) .+ (A*(-μν .+ Cm) + Q*Um), dims=2)
end

# Get the log probability of the sample point
function Distributions.logpdf(gp::GaussianProcess, y::Array{Float64})
    μ_vals = μ(gp.x, gp.k)
    Σ = K(gp.x, gp.x, gp.k)
    logpdf(MvNormal(μ_vals, Σ), y)
end

