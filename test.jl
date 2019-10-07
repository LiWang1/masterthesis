using DifferentialEquations, Optim, RecursiveArrayTools, DiffEqParamEstim
#using Plots

function f(du, u, p,t)
    du[1] = dx = p[1]*u[1] - u[1]*u[2]
    du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob = ODEProblem(f, u0, tspan, p)

sol = solve(prob, Tsit5())
t = collect(range(0, stop=10, length=200))
randomized = VectorOfArray([sol(t[i]) + 0.01randn(2) for i in 1:length(t)])
data = convert(Array, randomized)

cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t, data), maxiters=10000, verbose=false)
result = optimize(cost_function, 0.0, 10.0)
