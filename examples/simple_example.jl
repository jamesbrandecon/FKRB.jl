using FRAC, FKRB
using Distributions, Plots
using FixedEffectModels, DataFrames

T = 500; # (number of markets)/2
J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets
preference_means = [-1.0 1.0]; # means of the normal distributions of preferences -- 
# first column is preference for price, second is x
preference_SDs = [0.3 0.3]; # standard deviations of the normal distributions of preferences
sd_xi = 0.3; # standard deviation of demand shocks 

# Simulate mixed logit market-level data 
df = FKRB.sim_logit_vary_J(J1, J2, T, 1, 
    preference_means, preference_SDs, sd_xi, 
    with_market_FEs = true);
df = select(df, Not(:xi))

# If you don't provide domain ranges, define_problem will run FRAC.jl with the same problem specs
# and will use the estimated variance of preferences to define a grid width 
# that should cover 99% of the preference domain
problem = FKRB.define_problem( 
        data = df, 
        linear = ["x", "prices"], 
        nonlinear = ["x", "prices"], 
        iv = ["demand_instruments0"]);

problem = FKRB.define_problem( 
            data = df, 
            linear = ["x", "prices"], 
            nonlinear = ["x", "prices"], 
            range = Dict("x" => 0:0.1:2, "prices" => -2:0.1:0));

# One-line estimation
FKRB.estimate!(problem, 
    method = "elasticnet",
    gamma = 1e-5, # L1 penalty
    lambda = 1e-6) # L2 penalty

# ----------------
# Plots
# ----------------
# CDF
cdf_plot = plot_cdfs(problem)
# add true CDFs
plot!(cdf_plot, unique(problem.grid_points[:,1]), cdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,1])), color = :blue, ls = :dash, label = "CDF 1, true")
plot!(cdf_plot, unique(problem.grid_points[:,2]), cdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,2])), color = :red, ls = :dash, label = "CDF 2, true")

pmf_plot = FKRB.plot_pmfs(problem)
# add true PMFs 
plot!(pmf_plot, unique(problem.grid_points[:,1]), pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,1])) ./sum(pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,1]))), color = :blue, ls = :dash, label = "PDF 1, true")
plot!(pmf_plot, unique(problem.grid_points[:,2]), pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,2])) ./sum(pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,2]))), color = :red, ls = :dash, label = "PDF 2, true")

# -------------------------------
# Inference + post-estimation
# -------------------------------
FKRB.subsample!(problem; n_samples = 5)
FKRB.bootstrap!(problem, n_samples = 5)

# You can then pull the estimated weights and standard errors for individual regression coefficients: 
parameters = problem.results[1]; # parameters will contain the weights associated with each grid point
std = problem.std; # Only valid after running subsample! or boostrap!-- 
# std will contain the bootstrapped/subsampled standard error 
grid = rename(DataFrame(problem.grid_points, :auto), :x1 => :x, :x2 => :prices);

# Can calculate all price elasticities into a DataFrame  
FKRB.price_elasticities!(problem)
elasticities_df = FKRB.elasticities_df(problem)

own_elasticities = elasticities_df[elasticities_df.product1 .== elasticities_df.product2,:elast]
histogram(own_elasticities, label = "", xlabel = "Own-Price Elasticity", ylabel = "Count")
