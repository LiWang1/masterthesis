# Author: Li WANG
# Mail: wangli@student.ethz.ch
#
#
# --------------------------------------
using Pkg
Pkg.activate(".")

using Optim, ForwardDiff

# ------------
# 1) define true model and data generation
function y(x)
    return y = x[1]+2*x[2]+3*x[3]
end
x_obs = randn(2000, 3)
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
    G[1] = grad[1]
    G[2] = grad[2]
    G[3] = grad[3]
end


# 5) optimize the loss function with diffferent method
x0 = [0.0, 0.0, 0.0]
# without gradient
#res1 = optimize(loss, x0)
# autodiff
res2 = optimize(loss, grad!, x0, BFGS())
#res3 = optimize(loss, x0, Newton(); autodiff = :forward)
# numerical diff
#res4 = optimize(loss, x0, BFGS())
#res5 = optimize(loss, x0, Newton())
