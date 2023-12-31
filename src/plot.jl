using MeshPlotter
using PyPlot
MP = MeshPlotter

function plot_env_score!(ax, score; coords = (0.8, 0.8), fontsize = 50)
    tpars = Dict(
        :color => "black",
        :horizontalalignment => "center",
        :verticalalignment => "center",
        :fontsize => fontsize,
        :fontweight => "bold",
    )

    ax.text(coords[1], coords[2], score; tpars...)
end

function plot_env(
    _env, 
    score, 
    number_elements = false, 
    internal_order = false, 
    number_vertices = false,
    show_scores=true
    )

    mark_vertices = number_vertices ? findall(_env.mesh.active_vertex) : false
    element_numbers = number_elements ? findall(_env.mesh.active_triangle) : false

    env = deepcopy(_env)

    TM.reindex!(env)
    mesh = env.mesh
    if show_scores
        vs = TM.active_vertex_score(env)
    else
        vs = []
    end

    fig, ax = MP.plot_mesh(
        TM.active_vertex_coordinates(mesh),
        TM.active_triangle_connectivity(mesh),
        vertex_score=vs,
        vertex_size = 30,
        number_elements = element_numbers,
        internal_order = internal_order,
        number_vertices = mark_vertices
    )
    
    if show_scores
        plot_env_score!(ax, score)
    end

    return fig, ax
end

function plot_wrapper(
    wrapper; 
    filename = nothing,
    xlim=nothing,
    ylim=nothing,
    smooth_iterations = 5, 
    number_elements = false,
    number_vertices = false,
    show_scores=true
    )
    smooth_wrapper!(wrapper, smooth_iterations)

    text = string(wrapper.current_score) * " / " * string(wrapper.opt_score)

    internal_order = number_elements
    
    fig, ax = plot_env(
        wrapper.env, 
        text, 
        number_elements, 
        internal_order, 
        number_vertices,
        show_scores
    )

    if isnothing(xlim)
        ax.set_xlim(-1, 1)
    else
        ax.set_xlim(xlim...)
    end

    if isnothing(ylim)
        ax.set_ylim(-1, 1)
    else
        ax.set_ylim(ylim...)
    end

    if !isnothing(filename)
        fig.tight_layout()
        fig.savefig(filename)
    end

    return fig
end

function smooth_wrapper!(wrapper, num_iterations = 1)
    for iteration in 1:num_iterations
        TM.averagesmoothing!(wrapper.env.mesh)
    end
end


function plot_trajectory(policy, wrapper, root_directory)
    if !isdir(root_directory)
        mkpath(root_directory)
    end

    fig_name = "figure-" * lpad(0, 3, "0") * ".png"
    filename = joinpath(root_directory, fig_name)
    plot_wrapper(wrapper, filename=filename)

    fig_index = 1
    done = PPO.is_terminal(wrapper)
    while !done 
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        
        fig_name = "figure-" * lpad(fig_index, 3, "0") * ".png"
        filename = joinpath(root_directory, fig_name)
        plot_wrapper(wrapper, filename=filename)
        fig_index += 1

        done = PPO.is_terminal(wrapper)
    end
end
