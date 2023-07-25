using TOML
using BSON
include("../src/triangle_utilities.jl")
include("../src/random_environment_wrapper.jl")
include("../src/fixed_environment_wrapper.jl")
include("../src/plot.jl")


function initialize_environment(env_config)
    wrapper = initialize_fixed_environment(
        env_config["polygon_degree"],
        env_config["hmax"],
        env_config["allow_vertex_insert"],
        env_config["max_actions"]
    )
    return wrapper
end

function average_best_fixed_environment_returns(policy, num_trajectories, num_samples)
    ret = zeros(num_samples)
    for sample in 1:num_samples
        wrapper = initialize_environment(env_config)
        ret[sample] = best_normalized_best_return(policy, wrapper, num_trajectories)
    end
    return Flux.mean(ret), Flux.std(ret)
end

input_dir = "output/model-8"

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

env_config = config["environment"]

ret, dev = average_best_fixed_environment_returns(policy, 20, 100)

# wrapper = initialize_environment(env_config)
# plot_wrapper(wrapper)
# PPO.reset!(wrapper)
# ret = best_normalized_best_return(policy, wrapper, 100)
# plot_wrapper(wrapper)