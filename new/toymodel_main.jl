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
tspan_end = [30f0]

# ann

nn_1 = Chain(Dense(2, 10, tanh), Dense(10, 1, σ), Dense(1, 1))
nn_2 = Chain(Dense(2, 100, tanh), Dense(100, 1))
nn_3 = Chain(Dense(2, 1000, tanh), Dense(1000, 1))
nn_4 = Chain(Dense(2, 10000, tanh), Dense(10000, 1))
nn_5 = Chain(Dense(2, 100000, tanh), Dense(100000, 1))
ann = [nn_1]

# number of iterations
n_itr = [1, 10, 100, 1000, 10000]


# run the model
for i in n_itr
    for j in ann
        p,re = Flux.destructure(j)
        j_p = length(p)
        for k in tspan_end
            println("iter:", i, " ann: ", j_p, " tspan: ", k)
            println(@benchmark toy_model($i, $j, $k))
        end
    end
end

#
