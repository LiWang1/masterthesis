# batch for long term series
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
using CSV
using Interpolations
using Statistics
using Dates



include("batch_function_concise.jl")

# model parameters
p_ori = Float32[       50.0;  50.0;  50.0;  50.0;  50.0;  # k: storage constant
                       0.25;  0.25;  0.25;  0.25;  0.25;  # theta: runoff coefficient
                     186000; 62500; 66500; 44900;155800;  # area m^2
                       0.36;  0.12;  0.13;  0.09;   0.3;  # area ratio
                        650;   366;   898;   258;  1339;  # people eqivalent E
                        3]                                # infiltration L/s

# time span and time steps
batch_span = 7                       # 7 days of data
tspan_end = batch_span*1440f0 - 5f0
tspan = (0.0f0, tspan_end)
t = range(tspan[1], tspan[2], length = convert(Int64, floor(tspan_end)))

# u0
u_update = Float32[1.0; 1.0; 1.0; 1.0; 1.0]

# data input
f_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/carryonflow_all_cleaned.csv")
r_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/rain_all_cleaned.csv")
dwp = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/dwp_L_min_E.csv")

# data batches
# training
date_start_train= 20170608
date_end_train = addDays(date_start_train, batch_span-1)
batch_number = 1


flow_train = batch_sep("flow", f_03, date_start_train, date_end_train, batch_number, batch_span, t)
rain_train = batch_sep("rain", r_03, date_start_train, date_end_train, batch_number, batch_span, t)
dwp_train = dry(dwp, batch_number)

# testing
date_start_test= 20170607
date_end_test = addDays(date_start_test, batch_span-1)

flow_test = batch_sep("flow", f_03, date_start_test, date_end_test, 1, batch_span, t)
plot(flow_test)

rain_test = batch_sep("rain", r_03, date_start_test, date_end_test, 1, batch_span, t)
dwp_test = dry(dwp, 1)

# paras for NN
#ann = Chain(Dense(2, 10, σ), Dense(10, 10, σ), Dense(10, 1))
ann = Chain(Dense(2, 10, relu), Dense(10, 10, tanh),Dense(10, 10, tanh), Dense(10, 2, tanh), Dense(2, 1))
p1,re = Flux.destructure(ann)
n_itr = 250     # number of iteration for each batch
fig = 1         # fig == 1: visualize the training result

# tell me what you want to replace
rep = "dry"

# train the model with neural network
u, para = russikon_nn("dry",flow_train, rain_train, dwp_train, u_update, p1, re, tspan, t, n_itr, p_ori, fig)

# train the model with more data (minibatch)
#epoch = 2
#u, para = batchTrain(rep, epoch, flow_train, rain_train, dwp_train, ann, u_update, tspan, t, n_itr, p_ori, fig)
# should pass the rain data and dry flow inside


# evaluation of the model
orig = russikon_eval("original", flow_test, rain_test, dwp_test, u_update, p1, re, tspan, t, p_ori)
pred = russikon_eval(rep, flow_test, rain_test, dwp_test, u_update, para, re, tspan, t, p_ori)

#pred = russikon_eval(rep, flow_test, rain_test, dwp_test, u_update, p1, re, tspan, t, p_ori)
