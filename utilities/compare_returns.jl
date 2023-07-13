using BSON
include("../src/triangle_utilities.jl")
using PyPlot

function get_returns(input_dir)
    saved_data = joinpath(input_dir, "evaluator.bson")
    data = BSON.load(saved_data)[:data]
    evaluator = data["evaluator"]
    mean_returns = evaluator.mean_returns
    return mean_returns
end


mean_returns_1 = get_returns("output/model-1")
mean_returns_2 = get_returns("output/model-2")
mean_returns_3 = get_returns("output/model-3")

fig, ax = subplots()
ax.plot(mean_returns_2, label="w = 5e-4")
ax.plot(mean_returns_1, label="w = 1e-3")
ax.plot(mean_returns_3, label="w = 2e-3")
ax.grid()
ax.legend()
fig