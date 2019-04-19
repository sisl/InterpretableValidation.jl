using ExprRules

grammar = @grammar begin
    Real = Real * Real
    Real = _(Base.rand(1.0:5.0))  # special syntax, eval argument of _() at derivation time
end

rulenode = rand(RuleNode, grammar, :Real, 20)
display(rulenode, grammar)

Core.eval(rulenode, grammar)

