# Russikon model 
# Author: Li WANG
#
#
#----------------------------------------------
using Pkg
Pkg.activate(".")

using Interpolations
using CSV
using DifferentialEquations
using Plots
using Statistics
using Optim


include("function_nn_ode.jl")

# 1)read the input parameter
# parameters
#              B ;  C  ;  D  ;  E  ; F             # catchment index
p = Float32[    0;     0;    0 ;    0 ;   0  ;    # delayed mins
            100.0; 100.0; 100.0; 100.0; 100.0;    # k: storage constant
             0.31;  0.32;  0.35;  0.36;  0.36;    # theta: runoff coefficient
           186000; 62500; 66500; 44900;155800;    # area m^2
              650;   366;   898;   258;  1339;    # people eqivalent E
            200.0; 250.0;  90.0; 380.0;  75.0]    # overflow thresholds L/s

u0 =  Float32[1.0; 1.0; 1.0; 1.0; 1.0]            # initial conditions
tspan = (0.0f0, 1380.0f0)
t_eval = 0f0:60:1380f0                            # time for evaluation

# paras for nn
u0_ = param(u0)
ann = Chain(Dense(2, 10, relu), Dense(10, 10, relu), Dense(10, 1))
p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = param(p1)
ps = Flux.params(p2, u0_)

# rain
rain_data = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/rain_20170529_20170610.csv")
interp_linear_rain = LinearInterpolation(rain_data.mins, rain_data.V3)
rain(x) = convert(Float32, interp_linear_rain(x))

# dry weather pattern
dwp = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/dwp_L_min_E.csv")
interp_linear_dwp = LinearInterpolation(dwp.mins, dwp.L_mins)
dry(x) = convert(Float32, interp_linear_dwp(x%1440)/60)

# overflow_ reference data
overflow_data = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/carryon_20170529_20170610.csv")
interp_linear_overfow = LinearInterpolation(overflow_data.mins, overflow_data.V3)
overflow = convert(Array{Float32, 1}, interp_linear_overfow(t_eval))

# 2) calculate the carry on flow without neural network
sim_flow_wt = model_russikon(rain, dry, p, u0, tspan, t_eval)
loss = mean(abs2, overflow - sim_flow_wt)

# 3) Calculate the carry on flow with neural network
predict() = model_russikon2(rain, dry, p, u0_, tspan, t_eval, ann, p2)

# 4) training
# loss function
loss_adjoint() = mean(abs2, overflow - predict())
# algorithm for nn
data = Iterators.repeated((), 50) #number of iter.
opt = ADAM(0.1)
# visualization
cb()
# training
Flux.train!(loss_adjoint, ps, data, opt,  cb = cb)

cb = function ()
  display(loss_adjoint())
  display(loss)
  # plot current prediction against obs data
  pred = sim_flow_wt
  pred_nn = Flux.data(predict_adjoint())
  pl = scatter(t_eval, overflow, label = "obs")
  scatter!(pl, t_eval, pred_nn, label = "pred_nn")
  scatter!(pl, t_eval, pred, label = "pred")
  xlabel!("time_s")
  ylabel!("flow_L/s")
  display(plot(pl))
end
