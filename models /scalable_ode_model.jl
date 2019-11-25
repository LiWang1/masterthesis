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

#define the size
size_para = 2 #[1, 2, 4,   8,  16,  32,   64,    128]
              #[1, 4, 16, 64, 256, 1024, 4096, 16384] number of paras correps to size_para

# observation result
u0 = randn(size_para)
p = randn(size_para^2)
param_eval_gen = randn(size_para^2)
function scale_model(du, u, p, t)
  m = length(u)
  for i in 1:m
    du[i] = (p[m*(i-1)+1:m*i])'*u #u'
  end
end
tspan = (0.0,1.0)
prob = ODEProblem(scale_model,u0,tspan,p)
sol = solve(prob, Vern9())

# prediction with new paras
function test_f(p)
  _prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
  solve(_prob,Vern9())
end

# loss function
loss_nd(para) = sum((test_f(para)(sol.t) .- sol).^2)

# gradient for cost function to state
dg(out,u,p,t,i) = out .= 2*(sol(t).-u)

# gradient evaluation
rt_asa = minimum(@benchmark grad_asa= adjoint_sensitivities(test_f(param_eval_gen),Vern9(),dg,sol.t))
rt_fad = minimum(@benchmark grad_fad = ForwardDiff.gradient(loss_nd, param_eval_gen))
rt_fdm = minimum(@benchmark grad_fdm= Calculus.derivative(loss_nd, param_eval_gen))
rt_bad = minimum(@benchmark grad_bad = ReverseDiff.gradient(loss_nd, param_eval_gen))

println("asa: ", rt_asa)
println("fad: ", rt_fad)
println("bad: ", rt_bad)
println("fdm: ", rt_fdm)
