using MeshPlotter
using PlotQuadMesh
using RandomQuadMesh
using PyPlot
using TriMeshGame

TM = TriMeshGame
MP = MeshPlotter
PQ = PlotQuadMesh
RQ = RandomQuadMesh

function plot_polygon(boundary_points)
    boundary_points = [boundary_points boundary_points[:,1]]
    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(
        boundary_points[1,:], 
        boundary_points[2,:], 
        color="black",
        linewidth=2
    )
    return fig
end


boundary_points = RQ.random_polygon(20)
fig = plot_polygon(boundary_points)
output_file = "figures/plots/boundary.png"
fig.savefig(output_file)

mesh = RQ.tri_mesh(boundary_points, 
hmax=0.8, 
allow_vertex_insert=true
)

fig, ax = MP.plot_mesh(
    mesh.p,
    mesh.t
)
fig
output_file = "figures/plots/triangles.png"
fig.savefig(output_file)

q = zeros(Int, 4, 0)
quad_mesh = RQ.triquad_refine(mesh.p, q, mesh.t)
fig, ax = PQ.plot_mesh(quad_mesh.p, quad_mesh.t)
fig
output_file = "figures/plots/quad.png"
fig.savefig(output_file)