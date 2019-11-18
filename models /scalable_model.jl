# scalable model for
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


# model
function scale_model(du, u, p, t)
  for i in 1:4
    du[i] = transpose(p[4*i-3:4*i])*u
  end
end

p = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0]
u0 = [1.0, 0.0, 1.0, 0.0]
tspan = (0.8,1.0)
prob = ODEProblem(scale_model,u0,tspan, p)
sol = solve(prob)
plot(sol)

# predict result using new parameters
function test_f(p)
  _prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
  solve(_prob,Vern9(),save_everystep=false)[end]
end
function loss_nd(para)
    L = sum((test_f(para).- sol[end]).^2)
    return (L)
end

# gradient evaluation
p2 = [2.5, 2.0, 3.5, 4.0, 5.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0]

grad_fad = ForwardDiff.gradient(loss_nd, p2)
grad_fdm= Calculus.derivative(loss_nd, p2)
grad_bad = ReverseDiff.gradient(loss_nd, p2)
