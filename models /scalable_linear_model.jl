# Linear model
# Author: Li WANG
# Mail: wangli@student.ethz.ch
#
# --------------------------------------------------------------
using Pkg
Pkg.activate(".")

using Optim
using DifferentialEquations
using ForwardDiff
using Calculus
using ReverseDiff
using BenchmarkTools
using DiffEqSensitivity
using Plots


# ------------
num_paras = [1, 10, 100, 1000, 10000]# the number of parameters is num_paras^2

run_time_fad = Dict()
run_time_bad = Dict()
run_time_fdm = Dict()
run_time_asa = Dict()

for j in num_paras
    println(j)
    size_para = j
    param_gen = rand(size_para)
    param_eval_gen = rand(size_para)
    # 1.2) data generation for the parameters defined
    y(x::Vector) = param_gen'*x
    x_obs = randn(2000, size_para)
    y_obs = [y(x_obs[i,:]) + randn() for i in 1:2000]

    # predict with new parameter
    function predict_ln(x, para)
        solution = x*para
        return (solution)
    end

    # loss function
    function loss(para)
        y_est = predict_ln(x_obs, para)
        L = sum((y_est .- y_obs).^2)
        return (L)
    end

    p2 = param_eval_gen

    # gradient
    println(minimum(@benchmark ReverseDiff.gradient(loss, p2)))
    println(minimum(@benchmark ForwardDiff.gradient(loss, p2)))
    println(minimum(@benchmark Calculus.gradient(loss, p2)))
end
