# Linear model
# Author: Li WANG
# Mail: wangli@student.ethz.ch
#
# --------------------------------------------------------------
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ForwardDiff
using Calculus
using ReverseDiff
using Zygote
using BenchmarkTools
using DiffEqSensitivity



# ------------
#define the size
size_para = 256 #[1, 4, 16, 64, 256, 1024, 4096, 16384]

# observation result
param_gen = rand(size_para)
param_eval_gen = rand(size_para)
y(x::Vector) = param_gen'*x
x_obs = randn(2000, size_para)
y_obs = [y(x_obs[i,:]) + randn() for i in 1:2000]

# predict with new parameter
predict_ln(x, para) = x*para

# loss function
loss(para) = sum((predict_ln(x_obs, para) .- y_obs).^2)

# gradient evaluation
rt_fad = minimum(@benchmark grad_fad = ForwardDiff.gradient(loss, param_eval_gen))
rt_fdm = minimum(@benchmark grad_fdm= Calculus.derivative(loss, param_eval_gen))
rt_bad_r = minimum(@benchmark grad_bad = ReverseDiff.gradient(loss, param_eval_gen))
rt_bad_z = minimum(@benchmark grad_bad = Zygote.gradient(loss, param_eval_gen))


println("fad: ", rt_fad)
println("bad: ", rt_bad_r)
println("zyg: ", rt_bad_z)
println("fdm: ", rt_fdm)
