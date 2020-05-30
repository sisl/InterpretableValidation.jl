# Generate the default comparison range
function default_comparison_distribution(t::MvTimeseriesDistribution; σ_factor = 3)
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

# Sample a symbolic comparison
function sample_comparison(comparison_distribution::Dict{Symbol, Distribution}, rng::AbstractRNG = Random.GLOBAL_RNG)
    sym = rand(rng, keys(comparison_distribution))
    dist = comparison_distribution[sym]
    op = Symbol(".==")
    if dist isa ContinuousUnivariateDistribution
        op = rand(rng, [Symbol(".=="), Symbol(".<="), Symbol(".>=")])
    end
    val = rand(rng, dist)
    Expr(:call, op, sym, val)
end

# Function that returns the grammar for producing expressions of length N
function create_grammar()
    @grammar begin
        R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
        R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
        R = any_between(τ, C, C) | all_between(τ, C, C) # τ is true everywhere before or after C (inclusive)
        C = _(rand(InterpretableValidation.GRAMMAR_rng, 1:InterpretableValidation.GRAMMAR_N))
        τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
        τ = _(sample_comparison(InterpretableValidation.GRAMMAR_comparison_distribution, InterpretableValidation.GRAMMAR_rng))
    end
end

# Creates a standard loss function that samples trials and uses eval_fn to compute average loss
function loss_fn(eval_fn::Function, d::MvTimeseriesDistribution; rng::AbstractRNG = Random.GLOBAL_RNG, trials_per_expression = 10, max_loss = 1e9)
    function loss(rn::RuleNode, grammar::Grammar)
        ex = get_executable(rn, grammar)
        total_loss = 0.
        for i=1:trials_per_expression
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
        total_loss/trials_per_expression
    end
end

function set_global_grammar_params(N, comparison_distribution, rng)
    global GRAMMAR_N = N
    global GRAMMAR_comparison_distribution = comparison_distribution
    global GRAMMAR_rng = rng
end

# Wrapper for the optimization function
function optimize(eval_fn::Function,
                  d::MvTimeseriesDistribution;
                  rng::AbstractRNG = Random.GLOBAL_RNG,
                  loss = loss_fn(eval_fn, d, rng = rng),
                  Npop = 1000,
                  Niter = 30,
                  max_depth = 10,
                  opt = GeneticProgram(Npop, Niter, max_depth, 0.3, 0.3, 0.4),
                  comparison_distribution = default_comparison_distribution(d),
                  grammar = create_grammar(),
                  verbose = true
                 )
  # setup the global variables that the grammar uses
  set_global_grammar_params(N_pts(d), comparison_distribution, rng)
  ExprOptimization.optimize(opt, grammar, :R, loss, verbose = verbose)
end

