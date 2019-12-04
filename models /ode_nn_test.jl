# toy ode+NN model
# Author: Li WANG
#
#
#----------------------------------------------
using Pkg
Pkg.activate(".")

using DifferentialEquations
using DiffEqFlux, Flux, OrdinaryDiffEq

# true sol
function trueODEfunc(du,u,p,t)
    du[1] = 2*u[1] + 1*u[2]
    du[2] = 1*u[1] - 2*u[2]
end
u0 = Float32[.8; 0.8]
tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=30)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))


# ode + nn
u0_ = param(Float32[0.8; 0.8])
ann = Chain(Dense(2,10,tanh), Dense(10,1))

p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = param(p1)
ps = Flux.params(p2,u0_)

function dudt_(du,u,p,t)
    x, y = u
    du[1] = DiffEqFlux.restructure(ann,p[1:41])(u)[1]
    du[2] = 1*x - 2*y
end

prob_ = ODEProblem(dudt_,u0_,tspan,p2)
diffeq_adjoint(p2,prob_,Tsit5(),u0=u0_,abstol=1e-8,reltol=1e-6)

function predict_adjoint()
  diffeq_adjoint(p2,prob_,Tsit5(),u0=u0_,saveat=t)
end
loss_adjoint() = sum(abs2,ode_data - predict_adjoint())
loss_adjoint()

data = Iterators.repeated((), 200)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  # plot current prediction against data
  cur_pred = Flux.data(predict_adjoint())
  pl = scatter(t,ode_data[1,:],label="data_u1")
  scatter!(pl,t,ode_data[2,:],label="data_u2")
  scatter!(pl,t,cur_pred[1,:],label="prediction_u1")
  scatter!(pl,t,cur_pred[2,:],label="prediction_u2")
  display(plot(pl))
  #display(plot(solve(remake(prob,p=Flux.data(p3),u0=Flux.data(u0)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()
Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
