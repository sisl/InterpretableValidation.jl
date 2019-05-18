using Distributions
using LinearAlgebra
using Plots; gr()
using GaussianProcesses

μ(X, m) = [m(x) for x in X]
K(X, X′, k) = [k(x,x′) for x in X, x′ in X′]

sq_exp_kernl(x, x′, l = 1) = exp(-(x-x′)^2 / (2*l^2))
m(x) = 0

n=3
X = 2π * rand(n)
y = sin.(X)
L = 100
Xstar = range(0,stop=2π,length=L)
eye = 1e-6*Matrix{Float64}(I, n,n)

 K(X, X, k) + eye
μ_post(m, k, X, y, Xstar) = μ(Xstar,m) + K(Xstar, X, k) * ((K(X, X, k) + eye) \  (y .- μ(X,m)))

Σ_post(k, X, Xstar) = Matrix(Hermitian(K(Xstar, Xstar, k) - K(Xstar, X, k) * ((K(X, X, k) + eye) \ K(X, Xstar, k)) + 1e-6*Matrix{Float64}(I, L, L)))

function sample_gp(m, k, X, y, Xstar)
    N = MvNormal(μ_post(m, k, X, y, Xstar), Σ_post(k, X, Xstar))
    return rand(N)
end

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







