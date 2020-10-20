# Generate the default comparison range
function default_comparison_distribution(t::MvTimeseriesDistribution; σ_factor = 5)
    comparison_distribution = Dict{Symbol, Distribution}()
    for (sym, d) in t
        dist = d.timeseries_distribution
        if dist isa IID{Uniform{Float64}}
            d = dist.distribution
            comparison_distribution[sym] = Uniform(d.a, d.b)
        elseif dist isa IID{Normal{Float64}}
            d = dist.distribution
            comparison_distribution[sym] = Uniform(d.μ - σ_factor*d.σ, d.μ + σ_factor*d.σ)
        elseif dist isa IID{Categorical{Float64, Array{Float64,1}}}
            d = dist.distribution
            comparison_distribution[sym] = Categorical(length(d.p)) # make it uniformly probable
        elseif dist isa IID{Bernoulli{Float64}}
            comparison_distribution[sym] = DiscreteNonParametric([0,1], [0.5, 0.5]) # Make true false uniformly likely
        elseif dist isa GaussianProcess
            μ = mean([mean(rand(dist)) for i=1:50])
            σ = mean([std(rand(dist)) for i=1:50])
            comparison_distribution[sym] = Uniform(μ - σ_factor*σ, μ + σ_factor*σ)
        else
            throw(error("Unrecognized distribution type ", typeof(dist)))
        end
    end
    comparison_distribution
end

function default_comparisons(t::MvTimeseriesDistribution)
    comparisons = Dict{Symbol, Vector{Symbol}}()
    for (sym, d) in t
        if iscontinuous(d)
            comparisons[sym] = [Symbol(".<="), Symbol(".>=")]
        else
            comparisons[sym] = [Symbol(".==")]
        end
    end
    return comparisons
end

# Sample a symbolic comparison
function sample_comparison(comparison_distribution::Dict{Symbol, Distribution}, comparisons::Dict{Symbol, Vector{Symbol}}, rng::AbstractRNG = Random.GLOBAL_RNG)
    sym = rand(rng, keys(comparison_distribution))
    dist = comparison_distribution[sym]
    op = rand(rng, comparisons[sym])
    val = rand(rng, dist)
    Expr(:call, op, sym, val)
end

# Function that returns the grammar for producing expressions of length N
function create_stl_grammar(N, comparison_distribution, comparisons, rng = Random.GLOBAL_RNG)
    set_global_grammar_params(N, comparison_distribution, comparisons, rng)
    @grammar begin
        R = (R && R) | (R || R) | !R #| all(R) | any(R) | any_between(R, C, C) | all_between(R, C, C) # "and" and "or" expressions for scalar values
        R = all(τ) | any(τ) | any_between(τ, C, C) | all_between(τ, C, C) # τ is true everywhere before or after C (inclusive)
        C = _(rand(InterpretableValidation.GRAMMAR_rng, 1:InterpretableValidation.GRAMMAR_N))
        τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
        τ = _(sample_comparison(InterpretableValidation.GRAMMAR_comparison_distribution, InterpretableValidation.GRAMMAR_comparisons, InterpretableValidation.GRAMMAR_rng))
    end
end

function create_policy_grammar(N, disturbance_comparison_distribution, disturbance_comparisons, state_comparison_distribution, state_comparisons; rng = Random.GLOBAL_RNG)
    set_global_grammar_params(N,
                              disturbance_comparison_distribution,
                              disturbance_comparisons,
                              rng,
                              state_comparison_distribution = state_comparison_distribution,
                              state_comparisons = state_comparisons)
    @grammar begin
        R = R && R
        R = [S, X]
        S = (S .& S) | (S .| S)
        S = _(sample_comparison(InterpretableValidation.GRAMMAR_state_comparison_distribution, InterpretableValidation.GRAMMAR_state_comparisons, InterpretableValidation.GRAMMAR_rng))
        X = (X .& X) | (X .| X)
        X = _(sample_comparison(InterpretableValidation.GRAMMAR_comparison_distribution, InterpretableValidation.GRAMMAR_comparisons,  InterpretableValidation.GRAMMAR_rng))
    end
end

# Creates a standard loss function that samples trials and uses eval_fn to compute average loss
function loss_fn(eval_fn::Function, d::MvTimeseriesDistribution; rng::AbstractRNG = Random.GLOBAL_RNG, episodes = 10, max_loss = 1e9, length_penalty = 0.01)
    function loss(rn::RuleNode, grammar::Grammar)
        ex = get_executable(rn, grammar)
        total_loss = 0
        for i=1:episodes
            try
                timeseries = rand(rng, ex, d)
                total_loss += eval_fn(timeseries)
            catch e
                if e isa InfeasibleConstraint
                    total_loss += max_loss
                else
                    throw(e)
                end
            end
        end
        total_loss/episodes + length_penalty*length(rn)
    end
end

function get_policy(ex, s_fn::Function, x_fn::Function, d_fn::Function; rng::AbstractRNG = Random.GLOBAL_RNG)
    imps = parse_implications(ex)
    ks = collect(keys(imps))
    # Construct a policy to be used for rollouts
    function pol(s)
        d = d_fn(s)
        st = s_fn(s)
        valid_keys = ks[[interpret(st, k) for k in ks]]
        if length(valid_keys) == 0
            x = rand(rng, d)
        else
            constraint_expr = and_expressions([imps[k] for k in valid_keys])
            x = rand(rng, constraint_expr, d)
        end
        x_fn(x)
    end
    FunctionPolicy(pol)
end

function policy_loss_fn(mdp, eval_fn, s_fn::Function, x_fn::Function, d_fn::Function; rng::AbstractRNG = Random.GLOBAL_RNG, episodes = 10, max_loss = 1e9, length_penalty = 0.01)
    function loss(rn::RuleNode, grammar::Grammar)
        ex = get_executable(rn, grammar)
        pol = get_policy(ex, s_fn, x_fn, d_fn, rng = rng)

        total_loss = 0
        for i=1:episodes
            try
                total_loss += eval_fn(pol)
            catch e
                if e isa InfeasibleConstraint
                    total_loss += max_loss
                else
                    throw(e)
                end
            end
        end
        total_loss/episodes + length_penalty*length(rn)
    end
end

function set_global_grammar_params(N, comparison_distribution, comparisons, rng; state_comparison_distribution = Dict(), state_comparisons = Dict())
    global GRAMMAR_state_comparison_distribution = state_comparison_distribution
    global GRAMMAR_state_comparisons = state_comparisons
    global GRAMMAR_N = N
    global GRAMMAR_comparison_distribution = comparison_distribution
    global GRAMMAR_comparisons = comparisons
    global GRAMMAR_rng = rng
end

# Wrapper for the optimization function
function optimize_timed_stl(eval_fn::Function,
                  d::MvTimeseriesDistribution;
                  rng::AbstractRNG = Random.GLOBAL_RNG,
                  episodes = 10,
                  loss = loss_fn(eval_fn, d, rng = rng, episodes = episodes),
                  Npop = 1000,
                  Niter = 30,
                  max_depth = 4,
                  top_k_frac = 0.3,
                  opt = GeneticProgram(Npop, Niter, max_depth, 0.3, 0.3, 0.4, init_method = ExprOptimization.GeneticPrograms.RandomInit(), select_method = ExprOptimization.GeneticPrograms.TruncationSelection(Int(floor(top_k_frac*Npop)))),
                  grammar = create_stl_grammar(N_pts(d), default_comparison_distribution(d), default_comparisons(d), rng),
                  verbose = true
                 )
  # setup the global variables that the grammar uses
  ExprOptimization.optimize(opt, grammar, :R, loss, verbose = verbose)
end

function optimize_stl_policy(mdp,
                  eval_fn::Function,
                  s_fn::Function,
                  x_fn::Function,
                  d_fn::Function,
                  grammar::Grammar;
                  episodes = 10,
                  rng::AbstractRNG = Random.GLOBAL_RNG,
                  loss = policy_loss_fn(mdp, eval_fn, s_fn, x_fn, d_fn, rng = rng, episodes = episodes),
                  Npop = 1000,
                  Niter = 30,
                  max_depth = 4,
                  top_k_frac = 0.3,
                  opt = GeneticProgram(Npop, Niter, max_depth, 0.3, 0.3, 0.4, init_method = ExprOptimization.GeneticPrograms.RandomInit(), select_method = ExprOptimization.GeneticPrograms.TruncationSelection(Int(floor(top_k_frac*Npop)))),
                  verbose = true
                 )
    ExprOptimization.optimize(opt, grammar, :R, loss, verbose = verbose)
end

# Function to generate the MvTimeseriesDistribution and eval function for a mdp with discrete actions
function discrete_action_mdp(mdp, N::Int64; backup_policy = RandomPolicy(mdp), rng = Random.GLOBAL_RNG, use_prob = true, action_probability = (mdp, a) -> 1. / length(actions(mdp)))
    x = collect(1.:N)
    t = MvTimeseriesDistribution(:a => IID(x, Categorical([action_probability(mdp, a) for a in actions(mdp)])))
    f = function eval_fn(d::Dict{Symbol, Array})
            as = [actions(mdp)[d[:a]]...]
            r = simulate(RolloutSimulator(rng), mdp, PlaybackPolicy(as, backup_policy))
            (use_prob) ? -log(r) - logpdf(t, d) : -r
        end
    t, f
end

function continuous_action_mdp(mdp, a_dist::Dict{Symbol, Distribution}, N::Int64; backup_policy, create_actions_fn, rng = Random.GLOBAL_RNG, use_prob = true)
    x = collect(1.:N)
    t = MvTimeseriesDistribution(key => IID(x, a_dist[key]) for key in keys(a_dist))
    f = function eval_fn(d::Dict{Symbol, Array})
            as = create_actions_fn(d)
            r = simulate(RolloutSimulator(rng), mdp, PlaybackPolicy(as, backup_policy))
            ret = (use_prob) ? -log(r) - logpdf(t, d) : -r
            isinf(ret) ? 1e9 : ret
        end
    t, f
end

function sample_history(expr::Expr, t::MvTimeseriesDistribution, mdp, rng = Random.GLOBAL_RNG; backup_policy = RandomPolicy(mdp))
    d = rand(rng, expr, t)
    as = [actions(mdp)[d[:a]]...]
    simulate(HistoryRecorder(), mdp, PlaybackPolicy(as, backup_policy))
end

