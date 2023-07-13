using TOML
using BSON
include("../src/triangle_utilities.jl")
include("../src/environment_wrapper.jl")
include("../src/plot.jl")

function initialize_environment(env_config)
    allow_vertex_insert = get(env_config, "allow_vertex_insert", true)
    env = RandPolyWrapper(
        env_config["polygon_degree"],
        env_config["hmax"],
        env_config["max_actions"],
        allow_vertex_insert
    )
    return env
end

input_dir = "output/model-6"

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

env_config = config["environment"]
wrapper = initialize_environment(env_config)

# plot_wrapper(wrapper)

# ret, dev = average_normalized_returns(policy, wrapper, 100)
# ret, dev, actions = average_normalized_returns_and_action_stats(policy, wrapper, 100)


rollout = 1

PPO.reset!(wrapper)
output_dir = joinpath(input_dir, "figures", "rollout-" * string(rollout))
plot_trajectory(policy, wrapper, output_dir)
rollout += 1