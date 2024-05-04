# FKRB.jl

This repo implements the method introduced by Fox, Kim, Ryan, and Bajari (2009) (FKRB) for estimating random coefficient mixed logit demand models. This approach allows us to estimate the distribution of consumer preferences nonparametrically and through a simple elastic net regression, thereby avoiding some of the convergence and speed issues with empirical estimation of random coefficient models. The package is barebones, but it allows you to do the following with just a few lines of code: 

- Estimate a random coefficient logit model 
- Run a bootstrap to get standard errors on the resulting model estimates (i.e., estimated weights, conditional on the grid) 
- Calculate price elasticities at existing prices (todo: allow for counterfactual prices)
- Plot the (nonparametric) distribution of random coefficients

Currently, I've only implemented the FKRB approach for market-level data. My goal is to have an API which is familiar and relatively consistent across [FRAC.jl](github.com/jamesbrandecon/FRAC.jl), [NPDemand.jl](github.com/jamesbrandecon/NPDemand.jl), and any other code I share for demand estimation, so that multiple packages can be tested quickly and eventually the packages can be merged together. 

## Installation 
Install from Github: 
```julia
using Pkg; Pkg.add(url = "https://github.com/jamesbrandecon/FKRB.jl")
```

## Usage 
```julia
using FKRB
T = 500; # number of markets/2
J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets
preference_means = [-1.0 1.0]; # means of the normal distributions of preferences -- first column is price, second is x
preference_SDs = [0.3 0.3]; # standard deviations of the normal distributions of preferences
sd_xi = 0.3; # standard deviation of demand shocks 

# Helper function to simulate mixed logit data with differing numbers of products across markets
df = FKRB.sim_logit_vary_J(J1, J2, T, 1, 
    preference_means, preference_SDs, sd_xi, with_market_FEs = true);
```

Estimation is then straightforward: 
```julia
problem = FKRB.define_problem( 
        data = df, 
        linear = ["x", "prices"], 
        nonlinear = ["x", "prices"], 
        iv = ["demand_instruments0"]);

FKRB.estimate!(problem, 
    gamma = 0.0001, # L1 penalty param
    lambda = 1e-6) # L2 penalty param
```

## Price endogeneity
The `FKRB` approach is best justified when all product characteristics are exogenous, but we are often interested in settings where that is not the case. The most general approach I've seen to handle endogeneity within this estimation approach is that from Meeker (2021), so I implemented a slightly modified version of his approach. I first estimate the model using `FRAC.jl`, using the same problem specifications that have been provided to `FKRB.jl`, meaning that the regression FRAC.jl uses allows for random coefficients and instrument for prices. Then, I store the estimated demand shocks from that procedure and include them in the market-level demand function when estimating through `FKRB.jl`. The intuition for this approach is that, by generating a good first estimate of demand shocks (which account for endogeneity), we can then control for those shocks in the second stage. If we think of the regression errors in the FKRB second stage as measurement error, then this approach avoids the omitted variable bias (endogeneity) induced by running `FKRB.jl` naively without correcting for endogenous prices. 

If you have an alternative preferred approach to estimating demand shocks, simply include a "xi" field in the data you provide to `define_problem` and these will be included in the utility function automatically. If you need fixed effects, estimate them first via FRAC or a logit spec, add them together with your residuals `xi`, and include the sum as the field "xi" in `problem.data`.

## Inference
I wasn't sure how to do inference here, but [Meeker (2021)](https://www.imeeker.com/files/jmp.pdf) recommends subsampling/bootstrapping. I've implemented two simple helper functions for this purpose: `bootstrap!` and `subsample!`, which implement the inference approaches corresponding to the function names. Both functions take the `FKRBProblem` as the sole required argument, plus keyword arguments controlling how many times to repeat the estimation procedure and how large a subsample to use. After one of these functions are called, `problem.std` will contain the implied standard errors from the called approach. 

## Visualization
I wrote two functions to plot the CDFs/PDFs (or PMFs, given the discreteness of the grid used in FKRB) of the estimated random coefficients: `plot_cdfs` and `plot_pmfs`.

```julia
pmf_plot = FKRB.plot_pmfs(problem)
# Add true PDFs from known distributions: 
plot!(pmf_plot, unique(problem.grid_points[:,1]), pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,1])) ./sum(pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,1]))), color = :blue, ls = :dash, label = "PDF 1, true")
plot!(pmf_plot, unique(problem.grid_points[:,2]), pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,2])) ./sum(pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,2]))), color = :red, ls = :dash, label = "PDF 2, true")
```