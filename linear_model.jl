# linear model
function y(x)
    return y = x[1]+2*x[2]+3*x[3]
end

# data generation
random = randn(2000, 3)
randomized = [y(random[i,:]) + randn() for i in 1:2000]

## define cost function and optimize
function solver(x, para)
    solution = x*para
    return (solution)
end

# cost function
function loss(para)
    obs = randomized
    ## solve model with the new parameter
    solution = solver(random, para)
    ## run model at time
    L = sum((solution .- obs).^2)
    return (L)
end

# optimize
using Optim
x0 = [0.0, 0.0, 0.0]
# without gradient
res1 = optimize(loss, x0)
# autodiff
res2 = optimize(loss, x0, BFGS(); autodiff = :forward)
res3 = optimize(loss, x0, Newton(); autodiff = :forward)
# numerical diff
res4 = optimize(loss, x0, BFGS())
res5 = optimize(loss, x0, Newton())
