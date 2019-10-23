# ODE model
# Author: Li WANG
# Mail: wangli@student.ethz.ch
#
#----------------------------------------------

using Pkg
Pkg.activate(".")

using DifferentialEquations, Optim, ForwardDiff

#batch reactor A + 2B <-> C, example 6.4 of SAMM
# 1) define the true model and data generation
function batch(du, u, p, t)
    du[1] = -p[1]*u[1]*u[2]*u[2]+p[2]u[3]
    du[2] = -2*p[1]*u[1]*u[2]*u[2]+2*p[2]*u[3]
    du[3] = p[1]*u[1]*u[2]*u[2]-p[2]*u[3]
end

u0 = [1.0, 2.0, 0.0]
p = [.5, .5]
tspan = (0.0, 1.0)
prob = ODEProblem(batch, u0, tspan, p)
sol = solve(prob, Tsit5())

#data generation using p=[1.0, 1.0]
x_obs = range(0, stop=1,length = 2000)
y_obs = [sol(x_obs[i]) + 0.01randn(3) for i in 1:length(x_obs)]

# 2) define the model
function solver(para)
    problem = ODEProblem(batch, u0, tspan, para)
    _problem = remake(problem;u0=convert.(eltype(para),problem.u0),p=para)
    solution = solve(_problem, Tsit5())
    return (solution)
end

# 3) cost function
function loss(para)
    solution = solver(para)
    L = 0.0
    for i in 1:length(x_obs)
        L += sum((solution(x_obs[i]) .- y_obs[i]).^2)
    end
    return (L)
end

# 4) gradient calculation
function grad!(G, para)
    grad = ForwardDiff.gradient(loss, para)
    for i in 1:length(para)
        G[i] = grad[i]
    end
end
# 5) hessian for Newton method 
function hess!(H, para)
    hess = ForwardDiff.hessian(loss, para)
    # here needs to be improved...
    H[1, 1] = hess[1, 1]
    H[1, 2] = hess[1, 2]
    H[2, 1] = hess[2, 1]
    H[2, 2] = hess[2, 2]
end

# 6) optimization
x0 = zeros(length(p))
res_wt_grad = optimize(loss, x0, iterations = 2000)
# newton works kind of well in this case
#res_grad = optimize(loss, x0, Newton(), autodiff = :forward)
res_grad = optimize(loss, grad!, hess!, x0, Newton())
