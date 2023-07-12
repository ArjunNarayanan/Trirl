using BSON
include("../src/triangle_utilities.jl")
using PyPlot


input_dir = "output/model-1/"

input_file = joinpath(input_dir, "loss.bson")
data = BSON.load(input_file)[:loss]

ppo_loss = data["ppo"]
entropy_loss = data["entropy"]

fig, ax = plt.subplots()
ax.plot(ppo_loss)
ax.grid()
ax.set_title("Loss history of PPO objective")
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("PPO Objective")
fig.tight_layout()
fig
output_file = joinpath(input_dir, "ppo_loss_history.png")
fig.savefig(output_file)

fig, ax = plt.subplots()
ax.plot(entropy_loss)
ax.grid()
ax.set_title("Loss history of entropy regularization")
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("-ve policy entropy")
fig.tight_layout()
fig
output_file = joinpath(input_dir, "entropy_loss_history.png")
fig.savefig(output_file)