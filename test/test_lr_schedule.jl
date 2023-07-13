using Flux
Optimiser = Flux.Optimise.Optimiser

function run_optimizer(optimizer)
    weights = [1.0]
    grads = [1.0]
    for iteration in 1:10
        lr = prod((opt.eta for opt in optimizer))
        println("Iteration $iteration | LR $lr")
        Flux.update!(optimizer, weights, grads)
    end
end


optimizer = Optimiser(
    Adam(1f0),
    ExpDecay(
        1.0,
        0.1,
        2
    )
)

run_optimizer(optimizer)