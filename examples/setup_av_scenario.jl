using ExprRules
using .LTLSampling

include("av_simulator.jl")

# Define the Simulator and action space
N = 40
x = 1:N
sim = AVSimulator()
car0 = Agent([-35,0], [11.17, 0])
peds0 = [Agent([-0.5,-2], [0,1])]
model = PedestrianModel(0.1, 0.1, 0.1)
A = ActionSpace(
        :ax => [-2, 2],
        :ay => [-2, 2],
        :nx => [-1, 1],
        :ny => [-1, 1],
        :nvx => [-1, 1],
        :nvy => [-1, 1])

range(-1,1, length=20)
range(-2,2, length=20)

# Define the grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all_before(τ, C) | all_after(τ, C) | all_between(τ, C, C) # τ is true everywhere before or after C (inclusive)
    C = |(1:40) # A random integer in the domain
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<=")))
    τ = _(sample_sym_comparison(A, Symbol(".>=")))
    τ = _(sample_sym_comparison(A, Symbol(".==")))
end
# τ = (ax .<= A) | (ax .>= A) | (ax .== A)
# τ = (ay .<= A) | (ay .>= A) | (ay .== A)
# τ = (nx .<= NR) | (nx .>= NR) | (nx .== NR)
# τ = (ny .<= NR) | (ny .>= NR) | (ny .== NR)
# τ = (nvx .<= NR) | (nvx .>= NR) | (nvx .== NR)
# τ = (nvy .<= NR) | (nvy .>= NR) | (nvy .== NR)
# A = |(-2:0.2222222222222222222:2)
# NR = |(-1:0.111111111111111111:1)

ex = get_executable(rand(RuleNode, grammar, :R), grammar)


# Define the loss function as the negative of a monte-carlo sampling of the reward
function mc_loss(tree::RuleNode, grammar::Grammar, complexity_param = 0)
    ex = get_executable(tree, grammar)
    trials = 10
    total_loss = 0
    for i=1:trials
        actions = []
        try
            actions = sample_series(ex, A, x, iid_samples)
        catch e
            if isa(e, InvalidExpression)
                return 1e9
            else
                println("uncaught error, ", e, " on expr: ", ex)
                error("Uncaught error, ", e)
            end
        end

        ts = create_actions(actions[:ax], actions[:ay], actions[:nx], actions[:ny], actions[:nvx], actions[:nvy])
        total_loss -= simulate(sim, ts, car0, peds0, model)[1]
    end

    total_loss/trials + complexity_param * count_nodes(tree)
end

