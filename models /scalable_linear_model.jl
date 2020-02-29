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
using Optim




# ------------

#define the size
size_para = 128#[1, 4, 16, 64, 256, 1024, 4096, 16384]
println("size: ", size_para)
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

function grad_fdm!(G, para)
    G .= Calculus.gradient(loss, para)
end

function hess_fdm!(H, para)
    H .= Calculus.hessian(loss, para)
end

function grad_fad!(G, para)
    G .= ForwardDiff.gradient(loss, para)
end

function hess_fad!(H, para)
    H .= ForwardDiff.hessian(loss, para)
end

function grad_bad!(G, para)
    G .= ReverseDiff.gradient(loss, para)
end

function hess_bad!(H, para)
    H .= ReverseDiff.hessian(loss, para)
end

x0 = rand(size_para)
#fad
println("bfgs_fad: ", minimum(@benchmark opt_bfgs_fad = optimize(loss, grad_fad!, x0, BFGS(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
println("nt_fad: ", minimum(@benchmark opt_nt_fad = optimize(loss, grad_fad!, hess_fad!, x0, Newton(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
#bad
println("bfgs_bad: ", minimum(@benchmark opt_bfgs_bad = optimize(loss, grad_bad!, x0, BFGS(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
println("nt_bad: ", minimum(@benchmark opt_nt_bad = optimize(loss, grad_bad!, hess_bad!, x0, Newton(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
#fdm
println("bfgs_fdm: ", minimum(@benchmark opt_bfgs_fdm = optimize(loss, grad_fdm!, x0, BFGS(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
println("nt_fdm: ", minimum(@benchmark opt_nt_fdm = optimize(loss, grad_fdm!, hess_fdm!, x0, Newton(), Optim.Options(show_trace=false, g_tol = 1e-2, iterations = 3000))))
