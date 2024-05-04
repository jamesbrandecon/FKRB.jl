module FKRB

# TODO 
# cross-validation 
# elasticities

# Import dependencies
using DataFrames, LinearAlgebra 
using IterTools
using Distributions, FRAC, Plots, FixedEffectModels
using SCS, Convex
using StatsBase, ForwardDiff

# using SCS, Convex, CSV,

# Include source files
include("core.jl")
include("estimation.jl")
include("simulation.jl")
include("visualization.jl")
include("grids.jl")
include("bootstrap.jl")
include("elasticities.jl")

export FKRBProblem, define_problem, estimate!, 
    generate_regressors_aggregate, make_grid_points, plot_cdfs,
    bootstrap!, subsample!, price_elasticities!

end
