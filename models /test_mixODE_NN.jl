using DiffEqFlux, Flux, OrdinaryDiffEq

u0 = param(Float32[0.8; 0.8])
tspan = (0.0f0, 25.0f0)
ann = Chain(Dense(2, 10, tanh), Dense(10, 1))

p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = Float32[-2.0, 1.1]
p3 = param([p1;p2])
ps = Flux.params(p3, u0)

function dudt_(du, u, p, t)
  x, y = u
  du[1] = DiffEqFlux.restructure(ann, p[1:41])(u)[1] # ?
  du[2] = -p[end-1]*y + p[end]*x
end

prob = ODEProblem(dudt_, u0, tspan, p3)

function predict_adjoint()
  diffeq_adjoint(p3, prob, Tsit5(), u0=u0, seveat = 0.0:0.1:25.0)
end

loss_adjoint() = sum(abs2, x-1 for x in predict_adjoint())

Flux.train!(loss_adjoint, ps, Iterators.repeated((), 50), ADAM(0.1))
