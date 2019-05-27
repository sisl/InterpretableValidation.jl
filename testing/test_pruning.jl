using ExprRules

grammar = @grammar begin
    R = (R && R)
    R = all(τ) | any(τ) | all_before(τ, C) | all_after(τ, C) | all_between(τ, C, C)
    C = |(1:50)
    τ = (τ .& τ)
    τ = (a .<= G) | (a .== G) | (a .>= G)
    G = |([a1, a2, a3])
    G = H
    H = |([b1, b2, b3])
    H = H + G
end

rn = rand(RuleNode, grammar, :R)
ex= get_executable(rn, grammar)

leaves = get_leaves_of_type(rn, grammar, :τ, [:C, :G, :H])

get_executable(prune(rn, leaves[2], grammar), grammar)

