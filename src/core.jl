# core.jl

export FKRBProblem, logexpratio, elasticnet, FKRBGridDetails
import Base.show 

"""
    logexpratio(x, β)

Calculate the log exponential ratio for the logit model.

Arguments:
- `x`: Matrix of covariates.
- `β`: Vector of coefficients.

Returns:
- Vector of log exponential ratios.
"""
function logexpratio(x, β)
    exp_xβ = exp.(x * β)
    return exp_xβ ./ (1 .+ sum(exp_xβ))
end

"""
    FKRBProblem

Struct representing the FKRB problem.

Fields:
- `data`: DataFrame containing the data.
- `linear`: Vector of strings representing linear variables.
- `nonlinear`: Vector of strings representing nonlinear variables.
- `iv`: Vector of strings representing instrumental variables.
- `grid_points`: Matrix of grid points for evaluating the FKRB model.
- `results`: Vector of weights (coefficients) estimated by the FKRB model.
- `train`: Vector of indices representing the training data (not implemented).
"""
mutable struct FKRBProblem
    data::DataFrame
    linear::Vector{String}
    nonlinear::Vector{String}
    iv::Vector{String}
    grid_points::Matrix{Float64}
    results
    train::Vector{Int}
    inference_results # raw results from bootstrap/subsampling
    std # standard errors on weights, reduced to a Vector
    all_elasticities
end

function Base.show(io::IO, problem::FKRBProblem)
    println("FKRB Problem: ")
    println("- Characteristics with random coefs: ", problem.nonlinear)
    println("- Number of markets: ", length(unique(problem.data.market_ids)))
    println("- Number of products: ", length(unique(problem.data.product_ids)))
    println("- Min products per market: ", minimum([length(unique(problem.data[problem.data.market_ids .== m, :product_ids])) for m in unique(problem.data.market_ids)]))
    println("- Max products per market: ", maximum([length(unique(problem.data[problem.data.market_ids .== m, :product_ids])) for m in unique(problem.data.market_ids)]))
    println("- Estimated: ", (problem.results!=[]))
end


mutable struct FKRBGridDetails
    ranges::Dict{String, StepRangeLen}
    method::String
end

""" 
    define_problem(;method = "FKRB", data=[], linear=[], nonlinear=[], iv=[], train=[])
Defines the FKRB problem, which is just a container for the data and results. 
"""
function define_problem(; 
    data=[], linear=[], nonlinear=[], iv=[], train=[],
    range = (-Inf, Inf), step = 0.1
    )
    # Any checks of the inputs
    # Are linear/nonlinear/iv all vectors of strings
    try 
        @assert eltype(linear) <: AbstractString
        @assert eltype(nonlinear) <: AbstractString
        @assert (eltype(iv) <: AbstractString) | (iv ==[])
    catch
        throw(ArgumentError("linear, nonlinear, and iv must be vectors of strings"))
    end

    # Are the variables in linear/nonlinear/iv all in data?
    try 
        @assert all([x ∈ names(data) for x ∈ linear])
        @assert all([x ∈ names(data) for x ∈ nonlinear])
        @assert all([x ∈ names(data) for x ∈ iv])
    catch
        throw(ArgumentError("linear, nonlinear, and iv must contain only variables present in data"))
    end

    # Do market_ids and product_ids exist? They should and they should uniquely identify rows 
    try 
        @assert "market_ids" ∈ names(data)
        @assert "product_ids" ∈ names(data)
        @assert size(unique(data, [:market_ids, :product_ids])) == size(data)
    catch
        throw(ArgumentError("Data should have fields `market_ids` and `product_ids`, and these should uniquely identify rows"))
    end

    if iv ==[]
        try 
            @assert (typeof(range) <: StepRangeLen) | (typeof(range) <: Dict)
        catch
            throw(ArgumentError("If `iv` is not provided, prices are assumed to be exogenous and custom ranges are required"))
        end
    end
    if (iv !=[])
        try 
            @assert ((range==(-Inf, Inf)) | ("xi" in names(data)))
        catch
            throw(ArgumentError("If `iv` is provided, ranges will be calculated automatically using FRAC.jl"))
        end
    end

    # Defind grid details object 
    if range == (-Inf, Inf)
        println("Using FRAC.jl to generate intiial guess for grid points......");
        instruments_in_df = (length(findall(occursin.(names(data), "demand_instruments")))>0);
        if instruments_in_df 
            println("Demand instruments detected -- using IVs for prices in FRAC.jl......")
        end
        frac_problem = FRAC.define_problem(data = data, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            fixed_effects = ["market_ids"],
            se_type = "robust", 
            constrained = false);

        FRAC.estimate!(frac_problem)

        data = frac_problem.data;

        range_dict = Dict()
        betas = coef(frac_problem.raw_results_internal)
        betanames = coefnames(frac_problem.raw_results_internal)
        alpha = 0.0001
        multiplier = quantile(Normal(0,1), 1-alpha/2);
        for nl in ["x", "prices"]
            ind_mean = findall(betanames .== nl);
            ind_sd = findall(betanames .== string("K_", nl));
            beta_mean = betas[ind_mean][1];
            beta_sd = abs(betas[ind_sd][1])
            nl_range = Base.range(beta_mean .- multiplier * sqrt(beta_sd), beta_mean .+ multiplier * sqrt(beta_sd), step = 0.1)
            push!(range_dict, nl => nl_range)
        end
    else
        if typeof(range) <: StepRangeLen
            range_dict = Dict{String, StepRangeLen}()
            for var in union(nonlinear)
                range_dict[var] = range
            end
        else
            try 
                @assert typeof(range) <: Dict
            catch
                throw(ArgumentError("range and step must be either a StepRangeLen or a Dict mapping variable names to ranges"))
            end
            range_dict = range
        end
    end
    grid_details = FKRBGridDetails(range_dict, "simple")
    grid_points = make_grid_points(data, linear, nonlinear; gridspec = grid_details);

    # Return the problem
    problem = FKRBProblem(sort(data, [:market_ids, :product_ids]), 
            linear, nonlinear, iv, grid_points, [], train, [], [], [])

    return problem
end

""" 
    generate_regressors_aggregate(problem::FKRBProblem)
Generates the RHS regressors for the version of the FKRB estimator that uses aggregate data.
"""
function generate_regressors_aggregate(problem; method = "level")
    # Unpack 
    data = problem.data
    linear = problem.linear;
    nonlinear = problem.nonlinear;
    all_vars = union(linear, nonlinear);
    grid_points = problem.grid_points;

    if method == "level"
        level_or_diff = (x,b) -> logexpratio(x, b);
    else
        level_or_diff_i = (x,b,i) -> ForwardDiff.gradient(x -> logexpratio(x,b)[i], x)[i,2];
        level_or_diff = (x,b) -> [level_or_diff_i(x,b,i) for i in 1:size(x,1)]
    end

    # Generate RHS regressors -- each term is a logit-form market share, calculated at a given grid point of parameters
    regressors = Array{Float64}(undef, size(data, 1), size(grid_points, 1)); 
    for m ∈ sort(unique(data.market_ids))
        i = 1;
        @views for g ∈ eachrow(grid_points) 
            X_m = Matrix(data[data.market_ids .== m, all_vars]);
            if "xi" in names(data)
                xi = data[data.market_ids .==m, :xi];
                regressors[data.market_ids .== m, i] = level_or_diff([X_m xi], vcat(g,1));
            else
                regressors[data.market_ids .== m, i] = level_or_diff(X_m, g);
            end
            i +=1;
        end
    end
    
    return regressors
end


""" 
    estimate!(problem::FKRBProblem; gamma = 0.3, lambda = 0.0)
Estimates the FKRB model using constrained elastic net. Problem is solved using Convex.jl, and estimated weights 
are constrained to be nonnegative and sum to 1. Results are stored in problem.results.
"""
function estimate!(problem::FKRBProblem; method = "elasticnet", gamma = 0.3, lambda = 0.0)
    data = problem.data;
    grid_points = problem.grid_points;

    if isempty(problem.train)
        train = 1:size(data, 1);
    else
        train = problem.train;
    end

    # Generate the RHS regressors that will show up in the estimation problem
    regressors = generate_regressors_aggregate(problem);

    # Combine into a DataFrame with one outcome and many regressors
    df_inner = DataFrame(regressors, :auto);
    df_inner[!,"y"] = data.shares;

    # Estimate the model
    # Simple version: OLS 
    if method == "ols"
        @views w = inv(Matrix(df[!,r"x"])' * Matrix(df[!,r"x"])) * Matrix(df[!,r"x"])' * Matrix(df[!,"y"])
    elseif method == "elasticnet"
        # Constrained elastic net: 
        # Currently for fixed user-provided hyperparameters, but could add cross-validation to choose them
        @views w_en = elasticnet(df_inner[!,"y"], df_inner[!,r"x"], gamma, lambda) 
        # MLJ version: @views w_en = elasticnet(df_inner[!,r"x"], df_inner[!,"y"]; gamma = gamma, lambda = lambda) 
    else
        throw(ArgumentError("Method `$method` not implemented -- choose between `elasticnet` or `ols`"))
    end

    problem.results = w_en;
end 