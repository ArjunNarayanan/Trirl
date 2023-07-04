using TOML

include("src/triangle_utilities.jl")
include("src/plot.jl")


function initialize_environment(env_config)
    env = RandPolyWrapper(
        env_config["polygon_degree"],
        env_config["hmax"],
        env_config["max_actions"],
    )
    return env
end

function initialize_policy(model_config)
    policy = DirectPolicy(
        model_config["input_channels"],
        model_config["hidden_channels"],
        model_config["num_hidden_layers"],
        model_config["output_channels"]
    )
    return policy
end



ARGS = ["output/model-1/config.toml"]
@assert length(ARGS) == 1 "Missing path to config file"
config_file = ARGS[1]
println("\t\tUSING CONFIG FILE : ", config_file)
config = TOML.parsefile(config_file)

wrapper = initialize_environment(config["environment"])
policy = initialize_policy(config["policy"]) |> gpu

optimizer = ADAM(1f-4)

evaluator_config = config["evaluator"]
default_outputdir = dirname(config_file)
output_dir = get(evaluator_config, "output_directory", default_outputdir)
num_evaluation_trajectories = evaluator_config["num_evaluation_trajectories"]
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

ppo_config = config["PPO"]
discount = Float32(ppo_config["discount"])
epsilon = Float32(ppo_config["epsilon"])
minibatch_size = ppo_config["minibatch_size"]
episodes_per_iteration = ppo_config["episodes_per_iteration"]
epochs_per_iteration = ppo_config["epochs_per_iteration"]
num_iter = ppo_config["number_of_iterations"]
entropy_weight = Float32(ppo_config["entropy"])

data_path = joinpath(output_dir, "data")

PPO.ppo_iterate!(
    policy,
    wrapper,
    optimizer,
    episodes_per_iteration,
    minibatch_size,
    num_iter,
    evaluator,
    epochs_per_iteration,
    discount,
    epsilon,
    entropy_weight,
    data_path
)