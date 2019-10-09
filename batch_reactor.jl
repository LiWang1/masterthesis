# used packages
using DifferentialEquations, Optim, RecursiveArrayTools, DiffEqParamEstim

#batch reactor A + 2B <-> C, example 6.4 of SAMM
function batch(du, u, p, t)
    du[1] = -p[1]*u[1]*u[2]*u[2]+p[2]u[3]
    du[2] = -2*p[1]*u[1]*u[2]*u[2]+2*p[2]*u[3]
    du[3] = p[1]*u[1]*u[2]*u[2]-p[2]*u[3]
end

u0 = [1.0, 2.0, 0.0]
p = [1.0, 1.0]
tspan = (0.0, 1.0)
prob = ODEProblem(batch, u0, tspan, p)
sol = solve(prob, Tsit5())

## data generation using p=[1.0, 1.0]
t = range(0, stop=1,length = 200)
randomized = VectorOfArray([sol(t[i]) + 0.01randn(3) for i in 1:length(t)])

## define cost function and optimize
function solver(para)
    problem = ODEProblem(batch, u0, tspan, para)
    solution = solve(problem, Tsit5())
    return (solution)
end

function sum_sqaure_arr(m)
    tot = 0.0
    for i in 1:length(m)
        tot += m[i]^2
    end
    return (tot)
end


# cost function
function loss(para)
    ## solve model with the new parameter
    solution = solver(para)
    ## run model at time
    L = 0.0
    for i in 1:length(t)
        L += sum_sqaure_arr(solution(t[i]) - randomized[i])
    end
    return (L)
end

#cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data,differ_weight=0.3, data_weight=0.7),
#                mixiters = 10000, verbose = false)
x0 = [0.0, 0.0]
res = Optim.optimize(loss, x0)
