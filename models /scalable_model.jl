# scalable model
# Author: Li WANG
#
#
#----------------------------------------------
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ForwardDiff
using Calculus
using ReverseDiff
using BenchmarkTools
using Plots

# Model
num_paras = 10 # the number of parameters is num_paras^2

run_time_fad = Dict()
run_time_bad = Dict()
run_time_fdm = Dict()

for j in 1:num_paras
  size_para = j
  param_gen = rand(size_para^2)
  param_eval_gen = rand(size_para^2)
  u0_gen = rand(size_para)

  function scale_model(du, u, p, t)
    for i in 1:size_para
      du[i] = transpose(p[size_para*(i-1)+1:size_para*i])*u #u'
    end
  end

  # Initiate a vec of length = 16
  #p = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0]
  p = param_gen
  u0 = u0_gen
  tspan = (0.0,1.0)
  prob = ODEProblem(scale_model,u0,tspan, p)
  sol = solve(prob)
  println(sol)
  #plot(sol)

  # Predict result using new parameters
  function test_f(p)
    _prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    solve(_prob,Vern9(),save_everystep=false)[end]
  end

  # Loss function
  function loss_nd(para)
      L = sum((test_f(para).- sol[end]).^2)
      return (L)
  end

  # Gradient evaluation
  #p2 = [2.5, 2.0, 3.5, 4.0, 5.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0]
  p2 = param_eval_gen
  rt_fad = @benchmark grad_fad = ForwardDiff.gradient(loss_nd, p2)
  rt_fdm = @benchmark grad_fdm= Calculus.derivative(loss_nd, p2)
  rt_bad = @benchmark grad_bad = ReverseDiff.gradient(loss_nd, p2)
  run_time_fad[j] = rt_fad
  run_time_fdm[j] = rt_fdm
  run_time_bad[j] = rt_bad
end

println(run_time_fad)
println(run_time_fdm)
println(run_time_bad)
