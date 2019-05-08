using ExprRules
include("../LTLSampling.jl")
include("../av_simulator.jl")

# Define the Simulator and action space
N = 50
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

# Define the grammar
grammar = @grammar begin
    R = (R && R) #| (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    R = all_before(τ, C) | all_after(τ, C) # τ is true everywhere before or after C (inclusive)
    C = _(rand(1:N)) # A random integer in the domain
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".<"))) # Sample a random less than comparison
    τ = _(sample_sym_comparison(A, Symbol(".>"))) # Sample a random greater than comparisonq
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end


# Define the loss function as the negative of a monte-carlo sampling of the reward
function mc_loss(tree::RuleNode, grammar::Grammar, complexity_param = 0)
    ex = get_executable(tree, grammar)
    trials = 10
    total_loss = 0
    for i in 1:trials
        action_ts = Dict()
        try
            action_ts = sample_series(ex, A, N)
        catch e
            return 1e9 # worst possible score if the expression is invalid
        end
        actions = create_actions(action_ts[:ax], action_ts[:ay], action_ts[:nx], action_ts[:ny], action_ts[:nvx], action_ts[:nvy])
        total_loss -= simulate(sim, actions, car0, peds0, model)[1]
    end
    total_loss/trials + complexity_param * count_nodes(tree)
end
