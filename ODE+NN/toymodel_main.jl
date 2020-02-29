# toy ode+NN model
# Author: Li WANG
#
#
#----------------------------------------------
using Pkg
Pkg.activate(".")

using DiffEqSensitivity, OrdinaryDiffEq, Zygote
using ForwardDiff
using Plots
using DiffEqFlux, Flux
using BenchmarkTools
using LinearAlgebra

include("toymodel_function.jl")

# tspan
tspan_end = [1000f0]

# ann
nn_1 = Chain(Dense(2, 10, relu), Dense(10, 1))
nn_2 = Chain(Dense(2, 10, relu), Dense(10, 10, relu), Dense(10, 1))
nn_3 = Chain(Dense(2, 10, relu), Dense(10, 10, relu), Dense(10, 10, relu), Dense(10, 1))
nn_4 = Chain(Dense(2, 10, relu), Dense(10, 10, relu), Dense(10, 10, relu), Dense(10, 10, relu), Dense(10, 1))
ann = [nn_1, nn_2, nn_3, nn_4]

# number of iterations
n_itr = [10]

#initial condition
init_u = Float32[5.0; 5.0]


# run the model
for i in n_itr
    for j in ann
        p,re = Flux.destructure(j)
        j_p = length(p)
        for k in tspan_end
            println("iter:", i, " ann: ", j_p, " tspan: ", k)
            println(@benchmark toy_model($i, $j, $k, $init_u))
        end
    end
end

#
