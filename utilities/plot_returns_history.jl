using BSON
include("../src/triangle_utilities.jl")
using PyPlot


input_dir = "output/model-5/"
output_dir = joinpath(input_dir, "figures")

if !isdir(output_dir)
    mkpath(output_dir)
end

saved_data = joinpath(input_dir, "evaluator.bson")
data = BSON.load(saved_data)[:data]
evaluator = data["evaluator"]

mean_returns = evaluator.mean_returns
print("Max performance : ", maximum(mean_returns))
dev = evaluator.std_returns

lower_bound = mean_returns - dev
upper_bound = mean_returns + dev

fig, ax = subplots()
ax.plot(mean_returns)
ax.fill_between(1:length(mean_returns),lower_bound, upper_bound, alpha = 0.4)
ax.grid()
ax.set_ylim([-1,1])
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean returns")
fig
# output_file = joinpath(output_dir, "returns.png")
# fig.savefig(output_file)