using InterpretableValidation
using Test
using Distributions
using Random

## IID tests
x = collect(1.:100.)
uniform  = IID(x, Uniform(2,3))
@test uniform isa IID{Uniform{Float64}}
normal = IID(x, Normal(0.5, 1.))
@test normal isa IID{Normal{Float64}}
categorical = IID(x, Categorical(5))
@test categorical isa IID{Categorical{Float64, Array{Float64,1}}}
bernoulli = IID(x, Bernoulli(0.1))
@test bernoulli isa IID{Bernoulli{Float64}}

@test N_pts(uniform) == 100

@test logpdf(uniform, rand(100)) == -Inf
@test logpdf(uniform, 2.5*ones(100)) == 0
@test logpdf(normal, ones(100)) ≈ 100*logpdf(Normal(0.5, 1.), 1.)
@test logpdf(categorical, ones(Int, 100)) ≈ 100*log(0.2)
@test logpdf(bernoulli, fill(true, 100)) ≈ 100*log(0.1)

v = rand(Random.GLOBAL_RNG, uniform)
@test all((v .> 2.) .& (v .< 3.))
v = rand(Random.GLOBAL_RNG, uniform, 2.5*ones(100), 2.7*ones(100))
@test all((v .> 2.5) .& (v .< 2.7))

v = rand(Random.GLOBAL_RNG, normal)
@test all((v .> -4.) .& (v .< 4.))
v = rand(Random.GLOBAL_RNG, normal, 2.5*ones(100), 2.7*ones(100))
@test all((v .> 2.5) .& (v .< 2.7))

v = rand(Random.GLOBAL_RNG, categorical)
@test all([v[i] in [1,2,3,4,5] for i=1:100])

v = rand(Random.GLOBAL_RNG, categorical, fill([1], 100))
@test all(v.==1)

v = rand(Random.GLOBAL_RNG, bernoulli)
@test sum(v) < 20

v = rand(Random.GLOBAL_RNG, bernoulli, fill([1], 100))
@test all(v)

## GaussianProcess Tests
gp = GaussianProcess(m = (x) -> 0, k = squared_exp_kernel(l=100), x = x)

@test gp.m(gp.x) == 0
@test gp.k(gp.x[1], gp.x[1]) == 1
@test gp.k(gp.x[1], gp.x[2]) < 1
@test gp.x == x
@test gp.σ2 == 1e-6

@test logpdf(gp, rand(100)) < 0

v = rand(Random.GLOBAL_RNG, gp)
@test length(v) == 100
l = collect(range(-1.5, 0, length=100))
u = collect(range(0, 1.5, length=100))
v = rand(Random.GLOBAL_RNG, gp, l, u)
ϵ = 0.18
@test all((v .>= l .- ϵ)) && all((v .<= u .+ ϵ))


