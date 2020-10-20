using InterpretableValidation
using Test

@test not_inv(:anybool, Random.GLOBAL_RNG) == :anybool
@test not_inv(true, Random.GLOBAL_RNG) == false
@test not_inv(false, Random.GLOBAL_RNG) == true


@test and_inv(true, Random.GLOBAL_RNG) == (true, true)
@test and_inv(false, Random.GLOBAL_RNG) in [(false, :anybool), (:anybool, false)]
@test and_inv(:anybool, Random.GLOBAL_RNG) == (:anybool, :anybool)

@test or_inv(true, Random.GLOBAL_RNG) in [(true, :anybool), (:anybool, true)]
@test or_inv(false, Random.GLOBAL_RNG) == (false, false)
@test or_inv(:anybool, Random.GLOBAL_RNG) == (:anybool, :anybool)

@test all_inv(true, 3, Random.GLOBAL_RNG) == ([true, true, true],)
@test all_inv(false, 2, Random.GLOBAL_RNG) in [([false, :anybool],), ([:anybool, false],)]
@test all_inv(:anybool, 3, Random.GLOBAL_RNG) == ([:anybool, :anybool, :anybool],)

@test any_inv(true, 2, Random.GLOBAL_RNG) in [([true, :anybool],), ([:anybool, true],)]
@test any_inv(false, 3, Random.GLOBAL_RNG) == ([false, false, false],)
@test any_inv(:anybool, 3, Random.GLOBAL_RNG) == ([:anybool, :anybool, :anybool],)

@test bitwise_and_inv([true, true], Random.GLOBAL_RNG) == ([true, true], [true, true])
@test bitwise_and_inv([:anybool, false], Random.GLOBAL_RNG) in [([:anybool, :anybool], [:anybool, false]),
                                    ([:anybool, false], [:anybool, :anybool])]

@test bitwise_or_inv([false, false], Random.GLOBAL_RNG) == ([false, false], [false, false])
@test bitwise_or_inv([:anybool, true], Random.GLOBAL_RNG) in [([:anybool, :anybool], [:anybool, true]),
                                        ([:anybool, true], [:anybool, :anybool])]


@test all_before([true, false, true], 2) == false
@test all_before([true, false, true], 1) == true
@test all_before_inv(true, 2, 3, Random.GLOBAL_RNG) == ([true, true, :anybool],)
@test all_before_inv(false, 2, 3, Random.GLOBAL_RNG)[1] in ([:anybool, false, :anybool], [false, :anybool, :anybool] )

@test all_after([true, false, true], 2) == false
@test all_after([true, false, true], 3) == true
@test all_after_inv(true, 2, 3, Random.GLOBAL_RNG) == ([:anybool, true, true],)
@test all_after_inv(false, 2, 3, Random.GLOBAL_RNG)[1] in ([:anybool, false, :anybool], [:anybool, :anybool, false] )


@test !all_between([false, true, true, false], 2, 4)
@test !all_between([false, true, true, false], 1, 3)
@test all_between([false, true, true, false], 2, 3)
@test all_between_inv(true, 2, 3, 4, Random.GLOBAL_RNG) == ([:anybool, true, true, :anybool],)
@test all_between_inv(false, 2, 3, 4, Random.GLOBAL_RNG)[1] in ([:anybool, false, :anybool, :anybool],[:anybool, :anybool, false, :anybool])

@test !any_between([true, false, false, false], 2, 4)
@test !any_between([false, true, true, false], 4, 4)
@test any_between([false, true, false, false], 2, 3)
@test any_between_inv(true, 2, 3, 4, Random.GLOBAL_RNG)[1] in ([:anybool, :anybool, true, :anybool], [:anybool, true, :anybool, :anybool])
@test any_between_inv(false, 2, 3, 4, Random.GLOBAL_RNG) == ([:anybool, false, false, :anybool],)



constraints = sample_constraints(Meta.parse("all(x .>= 1.) && any(x .>= 10)"), 10, Random.GLOBAL_RNG)
@test length(constraints) == 2
@test constraints[1][1] == Meta.parse("x .>= 1.")
@test constraints[1][2] == fill(true, 10)
@test constraints[2][1] == Meta.parse("x .>= 10")
@test sum(constraints[2][2] .== :anybool) == 9
@test sum(constraints[2][2] .== true) == 1


# Test nested temporal operators
expr = Meta.parse("any_between( all(x .>= 1.), 1,2)")
constraints = sample_constraints(expr, 10, Random.GLOBAL_RNG)


expr = Meta.parse("all( any_between(x .>= 1., 1, 3))")
x = collect(1:30.)
constraints = sample_constraints(expr, 30, Random.GLOBAL_RNG)
mvts = MvTimeseriesDistribution(:x => IID(x, Uniform(0,1)))
constrain_timeseries!(mvts,constraints)

