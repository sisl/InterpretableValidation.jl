using Test

include("LTLSampling.jl")

############## Inverse logic tests ######################
@test and_inv(true) == (true, true)
@test and_inv(false) in [(false, :anybool), (:anybool, false)]
@test and_inv(:anybool) == (:anybool, :anybool)

@test or_inv(true) in [(true, :anybool), (:anybool, true)]
@test or_inv(false) == (false, false)
@test or_inv(:anybool) == (:anybool, :anybool)

@test all_inv(true, 3) == ([true, true, true],)
@test all_inv(false, 2) in [([false, :anybool],), ([:anybool, false],)]
@test all_inv(:anybool, 3) == ([:anybool, :anybool, :anybool],)

@test any_inv(true, 2) in [([true, :anybool],), ([:anybool, true],)]
@test any_inv(false, 3) == ([false, false, false],)
@test any_inv(:anybool, 3) == ([:anybool, :anybool, :anybool],)

@test bitwise_and_inv([true, true]) == ([true, true], [true, true])
@test bitwise_and_inv([:anybool, false]) in [([:anybool, :anybool], [:anybool, false]),
                                    ([:anybool, false], [:anybool, :anybool])]

@test bitwise_or_inv([false, false]) == ([false, false], [false, false])
@test bitwise_or_inv([:anybool, true]) in [([:anybool, :anybool], [:anybool, true]),
                                        ([:anybool, true], [:anybool, :anybool])]

@test bool_inverses[Symbol("&&")] == and_inv
@test bool_inverses[Symbol("||")] == or_inv
@test bool_inverses[Symbol("any")] == any_inv
@test bool_inverses[Symbol("all")] == all_inv
@test bool_inverses[Symbol(".&")] == bitwise_and_inv
@test bool_inverses[Symbol(".|")] == bitwise_or_inv

@test Symbol("==") in terminals && :< in terminals && :< in terminals
@test :any in expanders && :all in expanders

#TODO: eval_conditional_tree testcases


############## Sampling and action-space tests ######################
A = ActionSpace(:x => [-1,1], :y => [0,10])

@test syms(A) == [:x, :y]
@test A.bounds[:x] == [-1,1] && A.bounds[:y] == [0,10]
@test A.not_equal[:x] == [] && A.not_equal[:y] == []

@test action_space_valid(A)

update_min!(A, :x, -2)
@test A.bounds[:x][1] == -1
update_min!(A, :x, 2)
@test A.bounds[:x][1] == 2

update_max!(A, :y, 20)
@test A.bounds[:y][2] == 10
update_max!(A, :y, 2)
@test A.bounds[:y][2] == 2

@test !action_space_valid(A)

A.bounds[:x] = [0.,0.]
@test action_space_valid(A)
push!(A.not_equal[:x], 0.)
@test !action_space_valid(A)

A = ActionSpace(:x1 => [-1,1], :x2 => [-1,1], :x3 => [-1,1], :x4 => [-1,1], :x5 => [-1,1], :x6 => [-1,1])

constraints = [ Pair(Meta.parse("x1 == 1"), true),
                Pair(Meta.parse("x2 == 2"), false),
                Pair(Meta.parse("x3 < 0.5"), true),
                Pair(Meta.parse("x4 < 0.5"), false),
                Pair(Meta.parse("x5 > 0.5"), true),
                Pair(Meta.parse("x6 > 0.5"), false)
                ]
constrain_action_space!(A, constraints)
@test A.bounds[:x1] == [1,1]
@test A.bounds[:x2] == [-1,1]
@test A.bounds[:x3] == [-1,0.5]
@test A.bounds[:x4] == [0.5,1]
@test A.bounds[:x5] == [0.5,1]
@test A.bounds[:x6] == [-1,0.5]

@test A.not_equal[:x2] == [2]

for i=1:1000
    @test uniform_sample([-1,1], [0]) > -1
    @test uniform_sample([-1,1], [0]) < 1
    @test uniform_sample([-1,1], [0]) != 0
    @test uniform_sample([1,1] , []) == 1
    a = sample_action(A)
    @test a[1] >= A.bounds[:x1][1] && a[1] <= A.bounds[:x1][2]
    @test a[2] >= A.bounds[:x2][1] && a[2] <= A.bounds[:x2][2]
    @test a[3] >= A.bounds[:x3][1] && a[3] <= A.bounds[:x3][2]
    @test a[4] >= A.bounds[:x4][1] && a[4] <= A.bounds[:x4][2]
    @test a[5] >= A.bounds[:x5][1] && a[5] <= A.bounds[:x5][2]
    @test a[6] >= A.bounds[:x6][1] && a[6] <= A.bounds[:x6][2]

    sym = rand(syms(A))
    b = sample_action(A, sym)
    @test b >= A.bounds[sym][1] && b <= A.bounds[sym][2]
end


@test sample_sym_comparison(A, :<).head == :call
@test sample_sym_comparison(A, :>).args[1] == :>
@test sample_sym_comparison(A, Symbol("==")).args[1] == Symbol("==")

