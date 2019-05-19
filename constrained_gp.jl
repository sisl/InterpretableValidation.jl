using Distributions
using LinearAlgebra
using Plots; gr()

include("mvrandn.jl")


μ(X, m) = [m(x) for x in X]
K(X, X′, k) = [k(x,x′) for x in X, x′ in X′]

k(x, x′, l = .1) = exp(-(x-x′)^2 / (2*l^2))
m(x) = 0


# m - The mean function
# k - The kernel
# y - observations
# Xs - predictions
# σ2 - Variance of error in observations
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

X = [0.0]
y = [0.0]

Xs = [-1.:0.05:1...]
Xν = Xs
l = Xs .+ 0.5
l[1:40] .-= 0.5
u = Xs .+ 0.5

a = sample_constrained_gp(m, k, X, y, Xs, Xν, l, u, n=10)

plot(Xs, a)
plot!(Xs, l, linewidth=2, linecolor="black")
plot!(Xs, u, linewidth=2, linecolor="black")
plot!(legend=:bottomright)



plot()
for i = 1:10
    plot!(Xstar, sample_gp(m, sq_exp_kernl, X, y, Xstar), label="")
end

scatter!(X, y)


  #regressors

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.,0.)
gp = GP(X,y,mZero,kern)

y_hat_theirs, _ = predict_y(gp, xstar)
plot!(xstar, y_hat_theirs, label="theirs")
# scatter!(x, y)







