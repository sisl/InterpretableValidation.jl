# Struct that defines a constrained timeseries distribution
mutable struct ConstrainedTimeseriesDistribution
    vartype::Symbol # :continuous, :discrete
    lb::Array{Float64,1} # Lowerbound for continuous variables
    ub::Array{Float64,1} # Upperbound for continous variables
    feasible::Array{Array{Int64,1},1} # Feasible set for discrete variables
    timeseries_distribution::TimeseriesDistribution
end

# Construct a ConstrainedTimeseriesDisribution from a IID{Uniform} distribution
function ConstrainedTimeseriesDistribution(d::IID{Uniform{Float64}})
    lb = d.distribution.a*ones(N_pts(d))
    ub = d.distribution.b*ones(N_pts(d))
    ConstrainedTimeseriesDistribution(:continuous, lb, ub, Array{Int64}[], d)
end

# Construct a ConstrainedTimeseriesDisribution from a IID{Normal} distribution
function ConstrainedTimeseriesDistribution(d::IID{Normal{Float64}})
    lb = -Inf*ones(N_pts(d))
    ub = Inf*ones(N_pts(d))
    ConstrainedTimeseriesDistribution(:continuous, lb, ub, Array{Int64}[], d)
end

# Construct a ConstrainedTimeseriesDisribution from a IID{Categorical} distribution
function ConstrainedTimeseriesDistribution(d::IID{Categorical{Float64, Array{Float64, 1}}})
    feasible = [collect(1:length(d.distribution.p)) for i=1:N_pts(d)]
    ConstrainedTimeseriesDistribution(:discrete, Float64[], Float64[], feasible, d)
end

# Construct a ConstrainedTimeseriesDisribution from a IID{Bernoulli} distribution
function ConstrainedTimeseriesDistribution(d::IID{Bernoulli{Float64}})
    feasible = [[0,1] for i=1:N_pts(d)]
    ConstrainedTimeseriesDistribution(:discrete, Float64[], Float64[], feasible, d)
end

# Construct a ConstrainedTimeseriesDisribution from a GaussianProcess distribution
function ConstrainedTimeseriesDistribution(d::GaussianProcess)
    lb = -Inf*ones(length(d.x))
    ub = Inf*ones(length(d.x))
    ConstrainedTimeseriesDistribution(:continuous, lb, ub, Array{Int64}[], d)
end

# Convert function for ease of construction
Base.convert(::Type{ConstrainedTimeseriesDistribution}, x::TimeseriesDistribution) =  ConstrainedTimeseriesDistribution(x)

# Number of points in the timeseries
N_pts(t::ConstrainedTimeseriesDistribution) = N_pts(t.timeseries_distribution)

# Check if the ConstrainedTimeseriesDistribution is over a discrete random variable
isdiscrete(d::ConstrainedTimeseriesDistribution) = d.vartype == :discrete

# Check if the ConstrainedTimeseriesDistribution is over a continuous random variable
iscontinuous(d::ConstrainedTimeseriesDistribution) = d.vartype == :continuous

# Checks the validity of the constraints
function isfeasible(d::ConstrainedTimeseriesDistribution)
    isdiscrete(d) ? all(length.(d.feasible) .> 0) : all(d.ub .>= d.lb)
end

# Sampler for ConstrainedTimeseriesDistribution
function Base.rand(rng::AbstractRNG, d::ConstrainedTimeseriesDistribution)
    isdiscrete(d) ? rand(rng, d.timeseries_distribution, d.feasible) : rand(rng, d.timeseries_distribution, d.lb, d.ub)
end

# Defines a Multivariate, constrained time series distribution
MvTimeseriesDistribution = Dict{Symbol, ConstrainedTimeseriesDistribution}

# Logpdf of a MvTimeseries sample according to the provided distribution
Distributions.logpdf(t::MvTimeseriesDistribution, y::Dict{Symbol, Array}) = sum([logpdf(t[sym].timeseries_distribution, y[sym]) for sym in keys(t)])

# Check the validity of the constraints
isfeasible(t::MvTimeseriesDistribution) = all(isfeasible.(values(t)))

# Get the number of points in the MVTimeseriesDistribution. Checks for consistency across variables
function N_pts(t::MvTimeseriesDistribution)
    Ns = N_pts.(values(t))
    @assert all([Ns[1] == Ns[i] for i=2:length(Ns)])
    Ns[1]
end

# Adjusts the lower and upper constraints according to a greater than comparison
function greaterthan!(d::ConstrainedTimeseriesDistribution, val::Float64, truthvals::AbstractArray)
    @assert iscontinuous(d)
    truthvals
    d.lb[truthvals .== true] .= max.(d.lb[truthvals .== true], val)
    d.ub[truthvals .== false] .= min.(d.ub[truthvals .== false], val)
end

# Adjusts the lower and upper constraints according to a less than comparison
lessthan!(d::ConstrainedTimeseriesDistribution, val::Float64, truthvals::AbstractArray) = greaterthan!(d, val, flex_not.(truthvals))

# Adjusts the lower and upper constraints according to an equality comparison
function continuous_equality!(d::ConstrainedTimeseriesDistribution, val::Float64, truthvals::AbstractArray)
    truthvals[truthvals .!= true] .= :anybool
    lessthan!(d, val, truthvals)
    greaterthan!(d, val, truthvals)
end

# Adjusts the feasible set according to an equality comparison
function discrete_equality!(d::ConstrainedTimeseriesDistribution, val::Int64, truthvals::AbstractArray)
    @assert isdiscrete(d)
    for i = 1:length(truthvals)
        val_index = findfirst(d.feasible[i] .== val)
        if truthvals[i] == true
            d.feasible[i] = !isnothing(val_index) ? [val] : Int64[]
        elseif truthvals[i] == false
            !isnothing(val_index) && deleteat!(d.feasible[i], val_index)
        end
    end
end

# Constrains the MvTimeseriesDistribution according to the expression and the corresponding truth
function constrain_timeseries!(t::MvTimeseriesDistribution, expr::Expr, truthvals::AbstractArray)
    head, sym, val = (expr.head == :call) ? expr.args : [expr.head, expr.args...]
    val, d = Meta.eval(val), t[sym]
    if isdiscrete(d) && head == Symbol(".==")
        discrete_equality!(d, val, truthvals)
    elseif iscontinuous(d) && head == Symbol(".==")
        continuous_equality!(d, val, truthvals)
    elseif iscontinuous(d) &&  head == Symbol(".<=")
        lessthan!(d, val, truthvals)
    elseif iscontinuous(d) && head == Symbol(".>=")
        greaterthan!(d, val, truthvals)
    else
        @error string("Unknown expression ", expr)
    end
end

# Constrains the time series for each constraint in the array of constraints
function constrain_timeseries!(t::MvTimeseriesDistribution, constraints)
    for (expr, truthvals) in constraints
        constrain_timeseries!(t, expr, truthvals)
    end
end

# Samples a time series fromt the provided MvTimeseriesDistribution
function Base.rand(rng::AbstractRNG, d::MvTimeseriesDistribution)
    results = Dict{Symbol, Array}()
    for (sym, const_dist) in d
        results[sym] = rand(rng, const_dist)
    end
    results
end

# Error type for infeasible constraint
struct InfeasibleConstraint <: Exception
    msg::String
end

# Printing for the error
Base.showerror(io::IO, e::InfeasibleConstraint) = print(io, "Infeasible Constraint:  ", e.msg)

# Tries to sample a time series that satisfies the expression from the provided MvTimeseriesDistribution
function Base.rand(rng::AbstractRNG, expr::Expr, d::MvTimeseriesDistribution; validity_trials = 2)
    N = N_pts(d)
    for i=1:validity_trials
        d2 = MvTimeseriesDistribution()
        for (k, v) in d
            d2[k] = ConstrainedTimeseriesDistribution(v.vartype, copy(v.lb), copy(v.ub), deepcopy(v.feasible), v.timeseries_distribution)
        end
        constraints = sample_constraints(expr, N, rng::AbstractRNG)
        constrain_timeseries!(d2, constraints)
        isfeasible(d2) && return rand(rng, d2)
    end
    throw(InfeasibleConstraint(string("Couldn't find feasible constraints for ", expr)))
end

