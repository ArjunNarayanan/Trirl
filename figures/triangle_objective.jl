using MeshPlotter
using RandomQuadMesh
using ProximalPolicyOptimization
# using PlotQuadMesh
# using PyPlot
# using TriMeshGame

PPO = ProximalPolicyOptimization
TM = TriMeshGame
MP = MeshPlotter
# PQ = PlotQuadMesh
RQ = RandomQuadMesh

include("../src/random_environment_wrapper.jl")
include("../src/plot.jl")

wrapper = RandPolyWrapper(20, 0.4, 10, true)
fig = plot_wrapper(wrapper, show_scores=false)
output_file = "figures/plots/blank-objective.png"
fig.savefig(output_file)