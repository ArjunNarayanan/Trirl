using Flux

struct DirectPolicy
    model
    hidden_channels
    num_hidden_layers
end

function DirectPolicy(in_channels, hidden_channels, num_hidden_layers, num_output)
    model = []
    push!(model, Dense(in_channels, hidden_channels, leakyrelu))
    for i in 1:num_hidden_layers-1
        push!(model, Dense(hidden_channels, hidden_channels, leakyrelu))
    end
    push!(model, Dense(hidden_channels, num_output))
    model = Chain(model...)

    DirectPolicy(model, hidden_channels, num_hidden_layers)
end

Flux.@functor DirectPolicy

function Base.show(io::IO, p::DirectPolicy)
    s = "Policy\n\t$(p.hidden_channels) channels\n\t$(p.num_hidden_layers) layers"
    println(io, s)
end

function (p::DirectPolicy)(state)
    return p.model(state)
end


#####################################################################################################################
# EVALUATING POLICY
function PPO.action_probabilities(policy::DirectPolicy, state)
    # @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE
    
    feature_matrix, action_mask = state.feature_matrix, state.action_mask
    logits = vec(policy(feature_matrix)) + action_mask
    p = softmax(logits)

    return p
end

function PPO.batch_action_probabilities(policy::DirectPolicy, state)
    # @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE

    feature_matrix, action_mask = state.feature_matrix, state.action_mask
    nf, nq, nb = size(feature_matrix)
    logits = reshape(policy(feature_matrix), :, nb) + action_mask
    probs = softmax(logits, dims=1)
    return probs
end
#####################################################################################################################
