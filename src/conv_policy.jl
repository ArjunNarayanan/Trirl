struct MeshBlock
    layer1
    layer2
    activation
    layernorm
    in_features
    function MeshBlock(in_features)
        layer1 = Dense(6*in_features, in_features)
        layer2 = Dense(6*in_features, in_features)
        activation = leakyrelu
        layernorm = LayerNorm(in_features)
        return new(layer1, layer2, activation, layernorm, in_features)
    end
end

Flux.@functor MeshBlock

function cycle_and_pair_features(feature_matrix, column_pairs)
    cf = TM.cycle_edges(feature_matrix)
    pf = TM.zero_pad(cf)[:, column_pairs]
    output = vcat(cf, pf)
    return output
end

function (m::MeshBlock)(feature_matrix, column_pairs)
    y = cycle_and_pair_features(feature_matrix, column_pairs)
    y = m.layer1(y)
    y = m.layernorm(y)
    y = m.activation(y)

    y = cycle_and_pair_features(y, column_pairs)
    y = m.layer2(y)
    y = m.layernorm(y)
    y = m.activation(y)

    y = y + feature_matrix
    return y
end


struct ConvPolicy
    in_features
    hidden_features
    out_features
    number_of_layers
    layernorm
    blocks
    input_linear_layer
    output_linear_layer
    function ConvPolicy(in_features, hidden_features, out_features, number_of_layers)
        input_linear_layer = Dense(in_features, hidden_features)
        layernorm = LayerNorm(hidden_features)
        blocks = [MeshBlock(hidden_features) for i in 1:number_of_layers]
        output_linear_layer = Dense(hidden_features, out_features)
        return new(
            in_features,
            hidden_features,
            out_features,
            number_of_layers,
            layernorm,
            blocks,
            input_linear_layer,
            output_linear_layer
        )
    end
end


Flux.@functor ConvPolicy

function Base.show(io::IO, p::ConvPolicy)
    s = "Policy\n\t$(p.hidden_features) channels\n\t$(p.number_of_layers) layers"
    println(io, s)
end

function (p::ConvPolicy)(feature_matrix, column_pairs)
    y = p.input_linear_layer(feature_matrix)
    y = p.layernorm(y)

    for layer in p.blocks
        y = layer(y, column_pairs)
    end

    output = p.output_linear_layer(y)
    return output
end


function PPO.action_probabilities(policy::ConvPolicy, state)
    feature_matrix, column_pairs = state.feature_matrix, state.column_pairs
    output = policy(feature_matrix, column_pairs)
    logits = vec(output) + state.action_mask
    probs = softmax(logits)
    return probs
end