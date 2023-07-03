include("src/triangle_utilities.jl")
include("src/plot.jl")


polygon_degree = 20
hmax = 0.2
max_actions = 10

wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)

# PPO.reset!(wrapper)
# is_valid_mesh(wrapper.env.mesh)

policy = Policy(2, 16, 3, 5)

data_path = "output/model-1"
rollouts = PPO.Rollouts(data_path)

PPO.collect_rollouts!(rollouts, wrapper, policy, 100, 0.95)