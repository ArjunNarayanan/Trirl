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

function plot_env(_env, score, number_elements = false, internal_order = false, number_vertices = false)
    mark_vertices = number_vertices ? findall(_env.mesh.active_vertex) : false
    element_numbers = number_elements ? findall(_env.mesh.active_triangle) : false

    env = deepcopy(_env)

    TM.reindex!(env)
    mesh = env.mesh
    vs = TM.active_vertex_score(env)

    fig, ax = MP.plot_mesh(
        TM.active_vertex_coordinates(mesh),
        TM.active_triangle_connectivity(mesh),
        vertex_score=vs,
        vertex_size = 30,
        number_elements = element_numbers,
        internal_order = internal_order,
        number_vertices = mark_vertices
    )
    
    plot_env_score!(ax, score)

    return fig, ax
end

function plot_wrapper(
    wrapper; 
    filename = nothing,
    xlim=nothing,
    ylim=nothing,
    smooth_iterations = 5, 
    number_elements = false,
    number_vertices = false
    )
    smooth_wrapper!(wrapper, smooth_iterations)

    text = string(wrapper.current_score) * " / " * string(wrapper.opt_score)

    internal_order = number_elements
    
    fig, ax = plot_env(wrapper.env, text, number_elements, internal_order, number_vertices)
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
