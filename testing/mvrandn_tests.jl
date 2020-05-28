using InterpretableValidation
using Test
using LinearAlgebra
using Statistics
using Random

val = mvrandn(Random.GLOBAL_RNG, [0.5,0.5,0.5], [1,1,1,], 0.2*Array{Float64}(I, 3, 3), 100)
@test all(val .> 0.5) && all(val .<1.0)

val2 = mvrandn_μ(Random.GLOBAL_RNG,[0.75, 0.75, 0.75],  [0.5,0.5,0.5], [1,1,1,], 0.2*Array{Float64}(I, 3, 3), 100)

@test mean(val) < mean(val2)
@test abs(mean(val2) - 0.75) < 0.1

