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


include("batch_function.jl")

# model parameters
p_ori = Float32[       30.0;  30.0;  30.0;  30.0;  30.0;  # k: storage constant
                       0.31;  0.32;  0.35;  0.36;  0.36;  # theta: runoff coefficient
                     186000; 62500; 66500; 44900;155800;  # area m^2
                       0.36;  0.12;  0.13;  0.09;   0.3;  # area ratio
                        650;   366;   898;   258;  1339;  # people eqivalent E
                        3]                                # infiltration L/s

# time span and time steps
tspan_end = 11515f0
tspan = (0.0f0, tspan_end)
t = range(tspan[1], tspan[2], length = convert(Int64, floor(tspan_end)))

# u0
u_update = Float32[1.0; 1.0; 1.0; 1.0; 1.0]

# data input
f_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/carryonflow_all.csv")
r_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/rain_all.csv")
dwp = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/dwp_L_min_E.csv")

# data batches
# training
date_start= 20170601
date_end = 20170608
batch_number = 2
batch_span = 8       # 8 days of data

flow_train = batch_sep_flow(f_03, date_start, date_end, batch_number, batch_span, t)
rain_train = batch_sep_rain(r_03, date_start, date_end, batch_number, batch_span)
dwp_train = dry(dwp, batch_number)

# testing
date_start_test= 20170617
date_end_test = 20170624

flow_test = batch_sep_flow(f_03, date_start, date_end, 1, batch_span, t)
rain_test = batch_sep_rain(r_03, date_start, date_end, 1, batch_span)
dwp_test = dry(dwp, 1)

# paras for NN
ann = Chain(Dense(2, 10, σ), Dense(10, 10, σ), Dense(10, 1))
p1,re = Flux.destructure(ann)
n_itr = 10  # number of iteration for each batch
fig = 1  # fig == 1: visualize the training result

# train the model with neural network
russikon_nn(carryon1, rain1, dry, u_update, p1, re, tspan, t, n_itr, p_ori, fig)

# train the model with more data (minibatch)
u, para = batchTrain(Flow_train, Rain_train, Dry_train, ann, u_update, tspan, t, n_itr, p_ori, fig) # should pass the rain data and dry flow inside

# evaluation of the model
evaluation(carryon3, rain3, dry, u, para, re, tspan, t, p_ori)

# performance of the original model
russikon_original(carryon1, rain1, dry, u_update, tspan, t, p_ori)

# visualization and comparison