# sensitivity analysis for conceptual model
include("batch_function_concise.jl")
# data input
f_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/carryonflow_all_cleaned.csv")
r_03 = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/rain_all_cleaned.csv")
dwp = CSV.read("/Users/wangli/Desktop/masterthesis/simdata/dwp_L_min_E.csv")

sens = Dict()
for i in 1:29
  batch_span = 29
  tspan_end = batch_span*1440f0 - 5f0
  tspan = (0.0f0, tspan_end)
  ts= range(tspan[1], tspan[2], length = convert(Int64, floor(tspan_end)))

  # data batch
  date_start_test= 20170601# 20170600+i
  date_end_test = addDays(date_start_test, batch_span-1)

  #flow = batch_sep("flow", f_03, date_start_test, date_end_test, 1, batch_span, ts)
  rain = batch_sep("flow", r_03, date_start_test, date_end_test, 1, batch_span, ts)
  #dp = dry(dwp, 1)


  function russikon_sen(du, u, p, t)
    k = p[1:5]           # k: storage constant
    theta = p[6:10]      # theta: runoff coefficient
    infiltration = p[11] # infiltration
    # B
    du[1] = (theta[1] * 186000f0 * rain(t) + 650f0*dp(t)- u[1] + 0.36f0*infiltration)/k[1]
    # C
    du[2] = (theta[2] * 62500f0 * rain(t) + 366f0*dp(t) - u[2] + 0.12f0*infiltration)/k[2]
    # D
    du[3] = (theta[3] * 66500f0* rain(t) + 898f0*dp(t) - u[3] + 0.13f0*infiltration)/k[3]
    # E
    du[4] = (theta[4] * 44900f0 * rain(t) + 258f0*dp(t) - u[4] + 0.09f0*infiltration)/k[4]
    # F
    du[5] = (theta[5] * 155800f0 * rain(t) + 1339f0*dp(t) - u[5] + 0.3f0*infiltration)/k[5]
  end

  p =  Float32[ 50.0;  50.0;  50.0;  50.0;  50.0;  # k: storage constant
                0.25;  0.25;  0.25;  0.25;  0.25;  # theta: runoff coefficient
                 3.0]                              # infiltration L/s
      # 186000; 62500; 66500; 44900;155800;  # area m^2
        # 0.36;  0.12;  0.13;  0.09;   0.3;  # area ratio
        #  650;   366;   898;   258;  1339;  # people eqivalent E

  u0 = Float32[1.0; 1.0; 1.0; 1.0; 1.0]

  prob = ODEProblem(russikon_sen,u0,tspan,p)
  sol = concrete_solve(prob,Tsit5(),u0, p, saveat=ts; sensealg=TrackerAdjoint())

  dp1 = Zygote.gradient((p)->mean(concrete_solve(prob,Tsit5(),u0,p,saveat=ts,sensealg=TrackerAdjoint())),p)
  sens[i] = dp1
end

test = sens[1][1]
