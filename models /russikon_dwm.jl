# Russikon model
# Author: Li WANG
#
#
#----------------------------------------------
using Pkg
Pkg.activate(".")

using DifferentialEquations
using DiffEqFlux, Flux, OrdinaryDiffEq
using Plots
using Interpolations
using CSV

# read data
#rain_data = CSV.read("/Users/wangli/Desktop/rain_intensity_mm_min_0625.csv")
overflow_data = CSV.read("/Users/wangli/Desktop/carryon_flow_dwp.csv")
dwp = CSV.read("/Users/wangli/Desktop/dwp_L_min_E.csv")

# interpolate data
#interp_linear_rain = LinearInterpolation(rain_data.V2, rain_data.V3)
interp_linear_dwp = LinearInterpolation(dwp.min, dwp.L_min)
interp_linear_overfow = LinearInterpolation(overflow_data.mins, overflow_data.V3)



# dry weather pattern L/s
dryflow(x) = Float32(interp_linear_dwp(x%1440)/60)
# rain intensity mm/s
# rain_intensity(x) = 0.0f0#Float32(interp_linear_rain(x)/60)
# overflow L/s
t = collect(1:7199)   #saveat
overflow = convert(Array{Float32, 1}, Array(interp_linear_overfow(t)))

PT_B = 650f0 # E: equivalent
PT_C = 366f0
PT_D = 898f0
PT_E = 258f0
PT_F = 1339f0

theta_B = 0.31f0
theta_C = 0.32f0
theta_D = 0.35f0
theta_E = 0.36f0
theta_F = 0.36f0

area_cat_B = 186000f0 # m^2
area_cat_C = 62500f0
area_cat_D = 66500f0
area_cat_E = 44900f0
area_cat_F = 155800f0

function catchment(du, u, p, t)
  # B
  du[1] = theta_B * area_cat_B * rain_intensity(t) + PT_B*dryflow(t) - p[1] * u[1]
  # C
  du[2] = theta_C * area_cat_C * rain_intensity(t) + PT_C*dryflow(t) - p[2] * u[2]
  # D
  du[3] = theta_D * area_cat_D * rain_intensity(t) + PT_D*dryflow(t) - p[3] * u[3]
  # E
  du[4] = theta_E * area_cat_E * rain_intensity(t) + PT_E*dryflow(t) - p[4] * u[4]
  # F
  du[5] = theta_F * area_cat_F * rain_intensity(t) + PT_F*dryflow(t) - p[5] * u[5]
end


p = Float32[0.65; 0.65; 0.65; 0.65; 0.65]
u0 = Float32[0.8; 0.8; 0.8; 0.8; 0.8]
tspan = (1.0f0, 7200.0f0)
prob = ODEProblem(catchment, u0, tspan, p)
sol = Array(solve(prob, saveat = t))

# add time of delay
ave_t_B = 709 # min
ave_t_C = 707
ave_t_D = 8
ave_t_E = 10
ave_t_F = 10

size = 5760
over_B = sol[1, ave_t_B:size+ave_t_B]
over_C = sol[2, ave_t_C:size+ave_t_C]
over_D = sol[3, ave_t_D:size+ave_t_D]
over_E = sol[4, ave_t_E:size+ave_t_E]
over_F = sol[5, ave_t_F:size+ave_t_F]

# overflow infractures
size_vec = ones(length(over_B))
over_B = min(over_B, 200.0*size_vec)
over_BC = min(over_B+over_C, 250.0*size_vec)
over_D = min(over_D, 90.0*size_vec)
over_E = min(over_E, 380.0*size_vec)
over_sim = min(over_BC+over_D+over_E+over_F, 75*size_vec)

# plot
m = 1:size+1
k = 1:size+1
n = 121:size+121 # adjust the utc time and local time

q_out = over_sim/0.65

plot(m,q_out)
plot!(n, interp_linear_overfow(k))
