include("../constrained_gp.jl")

k(x, x′, l = .1) = exp(-(x-x′)^2 / (2*l^2))
m(x) = 0

X = [0.0]
y = [0.0]

Xs = [-1.:0.05:1...]
Xν = Xs
l = zeros(length(Xs))
l[1:10] .= -1.5
l[11:20] = Xs[11:20] .- 0.5
l[31:41] .= 0.5
u = Xs .+ 0.5
u[31:41] .= 0.5


a = sample_constrained_gp(m, k, X, y, Xs, Xν, l, u, n=10)

plot(Xs, a, label="", title="Constrained Gaussian Process", xlabel="x", ylabel="y")
plot!(Xs, l, linewidth=2, linecolor="black", label = "Constraints")
plot!(Xs, u, linewidth=2, linecolor="black", label="")

