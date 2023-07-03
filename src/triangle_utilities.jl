using Flux
using RandomQuadMesh
using TriMeshGame
using Distributions: Categorical
using BSON
using Printf

# using Revise
using ProximalPolicyOptimization

RQ = RandomQuadMesh
TM = TriMeshGame
PPO = ProximalPolicyOptimization

const NUM_ACTIONS_PER_EDGE = 3 # flip, split, collapse
const NO_ACTION_REWARD = -2
const NUM_EDGES_PER_ELEMENT = 3

include("state.jl")
include("environment_wrapper.jl")
include("policy.jl")
include("step.jl")
