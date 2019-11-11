# Linear model
# Author: Li WANG
# Mail: wangli@student.ethz.ch
#
# --------------------------------------------------------------
using Pkg
Pkg.activate(".")

using Optim, ForwardDiff, BenchmarkTools


# ------------
# 1) define true model and data generation
# 1.1) define the number of dimensions
num_true_para = 1000
true_para = 1:num_true_para
# 1.2) data generation for the parameters defined
y(x::Vector) = transpose(true_para)*x
x_obs = randn(2000, num_true_para)
y_obs = [y(x_obs[i,:]) + randn() for i in 1:2000]

# 2) define the linear model
function solver(x, para)
    solution = x*para
    return (solution)
end

# 3) define the cost function
function loss(para)
    y_est = solver(x_obs, para)
    L = sum((y_est .- y_obs).^2)
    return (L)
end

# 4) define the way to calculate the gradients for the cost function
function grad!(G, para)
    grad = ForwardDiff.gradient(loss, para)
    for i in 1:num_true_para
        G[i] = grad[i]
    end
end


# 5) optimize the loss function with diffferent method
x0 = zeros(num_true_para)
# without gradient
res1 = optimize(loss, x0, iterations = 2000)
# autodiff
res2 = optimize(loss, grad!, x0, BFGS())
res3 = optimize(loss, x0, Newton(); autodiff = :forward)
# numerical diff
res4 = optimize(loss, x0, BFGS())
#res5 = optimize(loss, x0, Newton())

#@benchmark optimize(loss, x0, Newton(); autodiff = :forward)
#@benchmark optimize(loss, x0, BFGS())
