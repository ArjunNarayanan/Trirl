function get_desired_degree(initial_boundary_points, vertex_on_boundary)
    num_verts = length(vertex_on_boundary)
    num_initial_boundary_points = size(initial_boundary_points, 2)
    initial_boundary_points_desired_degree = TM.get_desired_degree.(
        TM.get_polygon_interior_angles(initial_boundary_points)
        )

    initial_vertex_on_boundary = falses(num_verts)
    initial_vertex_on_boundary[1:num_initial_boundary_points] .= true

    additional_vertex_on_boundary = (.!initial_vertex_on_boundary) .& vertex_on_boundary

    desired_degree = fill(6, num_verts)
    desired_degree[1:num_initial_boundary_points] .= initial_boundary_points_desired_degree
    desired_degree[additional_vertex_on_boundary] .= 4

    return desired_degree
end

function generate_random_game_environment(polygon_degree, hmax, allow_vertex_insert=true)
    boundary_pts = RQ.random_polygon(polygon_degree)
    mesh = RQ.tri_mesh(boundary_pts, hmax = hmax, allow_vertex_insert = allow_vertex_insert)
    mesh = TM.Mesh(mesh.p, mesh.t)
    
    vertex_on_boundary = TM.active_vertex_on_boundary(mesh)
    desired_degree = get_desired_degree(boundary_pts, vertex_on_boundary)

    env = TM.GameEnv(mesh, desired_degree)

    return env
end

function global_score(vertex_score)
    return sum(abs.(vertex_score))
end

function optimum_score(vertex_score)
    return abs(sum(vertex_score))
end

function check_terminated(score, opt_score, num_actions, max_actions)
    return (score <= opt_score) || (num_actions >= max_actions)
end

mutable struct RandPolyWrapper
    polygon_degree
    hmax
    num_actions
    max_actions
    env
    current_score
    opt_score 
    is_terminated 
    reward
    allow_vertex_insert
    function RandPolyWrapper(polygon_degree, hmax, max_actions, allow_vertex_insert=true)
        @assert max_actions > 0
        @assert polygon_degree > 3

        env = generate_random_game_environment(polygon_degree, hmax, allow_vertex_insert)
        num_actions = 0
        current_score = global_score(env.vertex_score)
        opt_score = optimum_score(env.vertex_score)
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        reward = 0

        new(
            polygon_degree, 
            hmax, 
            num_actions, 
            max_actions, 
            env, 
            current_score, 
            opt_score,
            is_terminated, 
            reward,
            allow_vertex_insert
        )
    end
end

function Base.show(io::IO, wrapper::RandPolyWrapper)
    println(io, "RandPolyWrapper")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper)
    env = generate_random_game_environment(
        wrapper.polygon_degree, 
        wrapper.hmax,
        wrapper.allow_vertex_insert
    )
    
    wrapper.env = env
    wrapper.current_score = global_score(env.vertex_score)
    wrapper.opt_score = optimum_score(env.vertex_score)
    wrapper.num_actions = 0
    wrapper.reward = 0
    wrapper.is_terminated = check_terminated(
        wrapper.current_score, 
        wrapper.opt_score, 
        wrapper.num_actions, 
        wrapper.max_actions
    )
end

function PPO.is_terminal(wrapper)
    return wrapper.is_terminated
end