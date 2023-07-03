#####################################################################################################################
struct StateData
    feature_matrix
    column_pairs
    optimum_return
    action_mask
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
end

function Flux.gpu(s::StateData)
    return StateData(
        gpu(s.feature_matrix), 
        gpu(s.column_pairs), 
        s.optimum_return,
        gpu(s.action_mask)
    )
end

function Flux.cpu(s::StateData)
    return StateData(
        cpu(s.feature_matrix), 
        cpu(s.column_pairs), 
        s.optimum_return,
        gpu(s.action_mask)
    )
end

function wrapper_requires_reindex(wrapper)
    mesh = wrapper.env.mesh
    return (TM.num_triangles(mesh) < mesh.new_triangle_pointer - 1) ||
           (TM.num_vertices(mesh) < mesh.new_vertex_pointer - 1)
end

function construct_action_mask(num_half_edges)
    action_mask = zeros(Float32, NUM_ACTIONS_PER_EDGE * num_half_edges)
    return action_mask
end

function PPO.state(wrapper)
    env = wrapper.env
    if wrapper_requires_reindex(wrapper)
        TM.reindex!(env)
    end

    connectivity = vec(TM.active_triangle_connectivity(env.mesh))
    pairs = TM.active_edge_pairs(env.mesh)
    num_half_edges = length(connectivity)
    @assert length(pairs) == num_half_edges
    pairs[pairs .<= 0] .= num_half_edges + 1
    

    vs = env.vertex_score[connectivity]
    vd = env.mesh.degrees[connectivity]
    feature_matrix = Float32.(vcat(vs', vd'))
    optimum_return = wrapper.current_score - wrapper.opt_score
    action_mask = construct_action_mask(num_half_edges)

    return StateData(
        feature_matrix,
        pairs,
        optimum_return,
        action_mask
    )
end
#####################################################################################################################
