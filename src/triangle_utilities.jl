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
include("direct_policy.jl")
include("step.jl")
include("checkpoints.jl")

function PPO.batch_advantage(state, returns)
    return returns ./ state.optimum_return
end

function PPO.number_of_actions_per_state(state)
    vs = state.feature_matrix
    am = state.action_mask
    @assert ndims(vs) == 3
    @assert ndims(am) == 2
    @assert size(vs, 2) * NUM_ACTIONS_PER_EDGE == size(am, 1)
    return size(am, 1)
end
