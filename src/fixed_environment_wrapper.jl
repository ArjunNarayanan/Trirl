function _vanilla_global_score(vertex_score)
    return sum(abs.(vertex_score))
end

function global_score(vertex_score)
    return _vanilla_global_score(vertex_score)
end

function optimal_score(vertex_score)
    return abs(sum(vertex_score))
end

function check_terminated(current_score, opt_score, num_actions, max_actions)
    return (num_actions >= max_actions) || (current_score <= opt_score)
end

mutable struct FixedMeshWrapper
    mesh0
    desired_degree
    num_actions
    max_actions
    env
    current_score
    opt_score
    is_terminated
    reward
    function FixedMeshWrapper(mesh, desired_degree, max_actions)
        @assert max_actions > 0

        mesh0 = deepcopy(mesh)
        d0 = deepcopy(desired_degree)

        env = TM.GameEnv(mesh, desired_degree)
        current_score = global_score(env.vertex_score)
        opt_score = optimum_score(env.vertex_score)
        reward = 0
        num_actions = 0
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        
        new(
            mesh0, 
            d0, 
            num_actions, 
            max_actions, 
            env, 
            current_score, 
            opt_score, 
            is_terminated, 
            reward
        )
    end
end

function initialize_fixed_environment(
    polygon_degree, 
    hmax, 
    allow_vertex_insert, 
    max_actions
    )

    boundary_pts = RQ.random_polygon(polygon_degree)
    mesh = RQ.tri_mesh(boundary_pts, hmax = hmax, allow_vertex_insert = allow_vertex_insert)
    mesh = TM.Mesh(mesh.p, mesh.t)
    
    vertex_on_boundary = TM.active_vertex_on_boundary(mesh)
    desired_degree = get_desired_degree(boundary_pts, vertex_on_boundary)

    wrapper = FixedMeshWrapper(mesh, desired_degree, max_actions)
    return wrapper
end

function Base.show(io::IO, wrapper::FixedMeshWrapper)
    println(io, "FixedMeshEnv")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper::FixedMeshWrapper)
    mesh = deepcopy(wrapper.mesh0)
    d0 = deepcopy(wrapper.desired_degree)
    wrapper.env = TM.GameEnv(mesh, d0)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.reward = 0
    wrapper.num_actions = 0
    wrapper.opt_score = optimal_score(wrapper.env.vertex_score)
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score,
        wrapper.num_actions, wrapper.max_actions)
end
