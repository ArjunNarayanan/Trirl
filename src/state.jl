#####################################################################################################################
struct StateData
    feature_matrix
    action_mask
    optimum_return
end

function Base.show(io::IO, s::StateData)
    num_features, num_half_edges = size(s.feature_matrix)
    println(io, "StateData")
    println(io, "\tNum Features : " * string(num_features))
    println(io, "\tNum Edges    : " * string(num_half_edges))
end

function Flux.gpu(s::StateData)
    return StateData(
        gpu(s.feature_matrix), 
        gpu(s.action_mask), 
        s.optimum_return,
    )
end

function Flux.cpu(s::StateData)
    return StateData(
        cpu(s.feature_matrix), 
        cpu(s.action_mask), 
        s.optimum_return,
    )
end

function wrapper_requires_reindex(wrapper)
    mesh = wrapper.env.mesh
    return (TM.num_triangles(mesh) < mesh.new_triangle_pointer - 1) ||
           (TM.num_vertices(mesh) < mesh.new_vertex_pointer - 1)
end

function action_mask_value(flag)
    if flag
        return -Inf32
    else
        return 0.0f0
    end
end

function construct_action_mask(template)
    requires_mask = mapslices(x -> all(x .== 0), template, dims=1)
    requires_mask = repeat(requires_mask, inner=(NUM_ACTIONS_PER_EDGE, 1))
    requires_mask = vec(requires_mask)
    mask = action_mask_value.(requires_mask)
    return mask
end

function level4_active_template(wrapper)
    @assert !wrapper_requires_reindex(wrapper)
    num_elements = TM.num_triangles(wrapper.env.mesh)
    active_half_edges = 1:NUM_EDGES_PER_ELEMENT*num_elements
    template = TM.make_level4_template(wrapper.env.mesh)
    active_template = template[:, active_half_edges]
    return active_template
end



function PPO.state(wrapper)
    env = wrapper.env
    if wrapper_requires_reindex(wrapper)
        TM.reindex!(env)
    end

    template = level4_active_template(wrapper)
    am = construct_action_mask(template)
    
    vertex_score = Float32.(TM.active_vertex_score(env))
    push!(vertex_score, 0.0f0)
    vertex_degree = Float32.(TM.active_vertex_degrees(env.mesh))
    push!(vertex_degree, 0.0f0)
    @assert length(vertex_score) == length(vertex_degree)

    missing_index = length(vertex_score)
    template[template.==0] .= missing_index
    vs = vertex_score[template]
    vd = vertex_degree[template]

    matrix = vcat(vs, vd)
    opt_return = Float32(wrapper.current_score - wrapper.opt_score)

    s = StateData(matrix, am, opt_return)

    return s
end
#####################################################################################################################



#####################################################################################################################

function pad_vertex_scores(vertex_scores_vector)
    num_half_edges = [size(vs, 2) for vs in vertex_scores_vector]
    max_num_half_edges = maximum(num_half_edges)
    num_new_cols = max_num_half_edges .- num_half_edges
    padded_vertex_scores = [TM.zero_pad(vs, nc) for (vs, nc) in zip(vertex_scores_vector, num_new_cols)]
    return padded_vertex_scores
end

function pad_action_mask(action_mask_vector)
    num_actions = length.(action_mask_vector)
    max_num_actions = maximum(num_actions)
    num_new_actions = max_num_actions .- num_actions
    padded_action_mask = [TM.pad(am, nr, -Inf32) for (am, nr) in zip(action_mask_vector, num_new_actions)]
    return padded_action_mask
end

function prepare_state_data_for_batching(state_data_vector)
    feature_matrix = [s.feature_matrix for s in state_data_vector]
    action_mask = [s.action_mask for s in state_data_vector]

    padded_feature_matrix = pad_vertex_scores(feature_matrix)
    padded_action_mask = pad_action_mask(action_mask)
    opt_return = [s.optimum_return for s in state_data_vector]

    state_data = [StateData(vs, am, opt) for (vs, am, opt) in zip(padded_feature_matrix, padded_action_mask, opt_return)]
    return state_data
end

function PPO.batch_state(state_data_vector)
    state_data_vector = prepare_state_data_for_batching(state_data_vector)

    feature_matrix = [s.feature_matrix for s in state_data_vector]
    am = [s.action_mask for s in state_data_vector]
    opt_return = [s.optimum_return for s in state_data_vector]

    batch_feature_matrix = cat(feature_matrix..., dims=3)
    batch_action_mask = cat(am..., dims=2)
    return StateData(batch_feature_matrix, batch_action_mask, opt_return)
end

#####################################################################################################################