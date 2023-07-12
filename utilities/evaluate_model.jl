using BSON
include("../src/triangle_utilities.jl")
include("../src/environment_wrapper.jl")
include("../src/plot.jl")

input_dir = "output/model-3"
data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data]
policy = data["policy"]
rollout = 1

wrapper = RandPolyWrapper(20, 0.4, 40)
plot_wrapper(wrapper)

ret, dev = average_normalized_returns(policy, wrapper, 100)

output_dir = joinpath(input_dir, "plots", "rollout-" * string(rollout))
plot_trajectory(policy, wrapper, output_dir)
rollout += 1