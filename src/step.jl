function PPO.reward(wrapper)
    return wrapper.reward
end

function index_to_action(index)
    actions_per_triangle = NUM_EDGES_PER_ELEMENT * NUM_ACTIONS_PER_EDGE

    triangle = div(index - 1, actions_per_triangle) + 1

    triangle_action_idx = rem(index - 1, actions_per_triangle)
    edge = div(triangle_action_idx, NUM_ACTIONS_PER_EDGE) + 1
    action = rem(triangle_action_idx, NUM_ACTIONS_PER_EDGE) + 1

    return triangle, edge, action
end

function action_space_size(env)
    nt = TM.num_triangles(env.mesh)
    return nt * NUM_EDGES_PER_ELEMENT * NUM_ACTIONS_PER_EDGE
end

function is_valid_mesh(mesh)
    return TM.all_active_vertices(mesh) &&
           TM.no_triangle_self_reference(mesh) &&
           TM.all_active_triangle_or_boundary(mesh)
end

function update_env_after_step!(wrapper)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.num_actions += 1
    wrapper.is_terminated = check_terminated(
        wrapper.current_score,
        wrapper.opt_score,
        wrapper.num_actions,
        wrapper.max_actions
    )
end

function step_wrapper!(wrapper, triangle_index, half_edge_index, action_type)
    env = wrapper.env
    previous_score = wrapper.current_score
    success = false

    # @assert TM.is_active_triangle(env.mesh, triangle_index) "Attempting to act on inactive triangle $triangle_index with action ($triangle_index, $half_edge_index, $action_type)"
    @assert action_type in 1:NUM_ACTIONS_PER_EDGE "Expected action type in 1:$NUM_ACTIONS_PER_EDGE, got type = $action_type"
    @assert half_edge_index in 1:NUM_EDGES_PER_ELEMENT "Expected edge in 1:$NUM_EDGES_PER_ELEMENT, got edge = $half_edge_index"
    @assert is_valid_mesh(wrapper.env.mesh) "Invalid state encountered, check the environment"

    if action_type == 1
        success = TM.step_flip!(env, triangle_index, half_edge_index)
    elseif action_type == 2
        success = TM.step_split!(env, triangle_index, half_edge_index)
    elseif action_type == 3
        success = TM.step_collapse!(env, triangle_index, half_edge_index)
    else
        error("Unexpected action type $action_type")
    end
    
    update_env_after_step!(wrapper)

    if success
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end
end

function PPO.step!(wrapper, action_index)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !wrapper.is_terminated "Attempting to step in terminated environment with action $action_index"

    triangle, edge, type = index_to_action(action_index)
    step_wrapper!(wrapper, triangle, edge, type)
end
