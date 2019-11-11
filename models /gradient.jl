# forward AD
using ForwardDiff
using Optim
f(x::Vector) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function grad!(G, m)
    grad = ForwardDiff.gradient(f, m)
    G[1] = grad[1]
    G[2] = grad[2]
end

initial_x = [0.0, 0.0]
res = Optim.minimizer(optimize(f, grad!, initial_x, BFGS()))
