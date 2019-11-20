# scalable ode model
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
using DiffEqSensitivity
using Plots

# Model
num_paras = 2 # the number of parameters is num_paras^2

run_time_fad = Dict()
run_time_bad = Dict()
run_time_fdm = Dict()
run_time_asa = Dict()
for j in 1:num_paras
  size_para = j
  param_gen = rand(size_para^2)
  param_eval_gen = rand(size_para^2)
  u0_gen = rand(size_para)

  function scale_model(du, u, p, t)
    for i in 1:size_para
      du[i] = (p[size_para*(i-1)+1:size_para*i])'*u #u'
    end
  end

  # true result
  p = param_gen
  u0 = u0_gen
  tspan = (0.0,1.0)
  prob = ODEProblem(scale_model,u0,tspan, p)
  sol = solve(prob, Vern9(),abstol=1e-10,reltol=1e-10)
  ts = sol.t
  plot(sol)

  # Predict result using new parameters
  function test_f(p)
    _prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    solve(_prob,Vern9())
  end

  # Loss function
  function loss_nd(para)
      L = sum((test_f(para)(ts) .- sol).^2)
      return (L)
  end

  # gradient for cost function to state
  function dg(out,u,p,t,i)
    out .= 2*(sol(t).-u)
  end

  # Gradient evaluation
  p2 = param_eval_gen
  rt_asa = @benchmark grad_asa= adjoint_sensitivities(test_f(p2),Vern9(),dg,ts)
  rt_fad = @benchmark grad_fad = ForwardDiff.gradient(loss_nd, p2)
  rt_fdm = @benchmark grad_fdm= Calculus.derivative(loss_nd, p2)
  rt_bad = @benchmark grad_bad = ReverseDiff.gradient(loss_nd, p2)
  run_time_fad[j] = rt_fad
  run_time_fdm[j] = rt_fdm
  run_time_bad[j] = rt_bad
  run_time_asa[j] = rt_asa
end

println(run_time_fad)
println(run_time_fdm)
println(run_time_bad)
println(run_time_asa)
