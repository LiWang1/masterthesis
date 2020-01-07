
## models without nns
function model_russikon(rain, dry, p, u0, tspan, t_eval)
  #rain:   rain intensity data (mm/s)
  #dry:    dry weather flow for each PT (L/s)
  #p:      parameters to build the model (delayed mins,
  #        storage constant, catmentsize, PT, threshold value...)
  #u0:     initial value (L/s)
  #tspan:  time span for solving the ODEs
  #t_eval: time steps for evaluations
  function catchment(du, u, p, t)
    k = p[6:10]
    theta = p[11:15]
    area = p[16:20]
    pt = p[21:25]
    # B
    du[1] = (theta[1] * area[1] * rain(t) + pt[1]*dry(t)- u[1])/k[1]
    # C
    du[2] = (theta[2] * area[2] * rain(t) + pt[2]*dry(t) - u[2])/k[2]
    # D
    du[3] = (theta[3] * area[3]* rain(t) + pt[3]*dry(t) - u[3])/k[3]
    # E
    du[4] = (theta[4] * area[4] * rain(t) + pt[4]*dry(t) - u[4])/k[4]
    # F
    du[5] = (theta[5] * area[5] * rain(t) + pt[5]*dry(t) - u[5])/k[5]
    end
  prob = ODEProblem(catchment, u0, tspan, p)
  sol = solve(prob, Tsit5(), saveat = t_eval)
  #add delay
  #out_delay = addDelay(sol, p[1:5], t_eval)
  #add overflow infras
  sim_flow = addOverflow(sol, p[26:30])
  return (sim_flow)
end

##models with nns
function model_russikon2(rain, dry, p_fix, u0, tspan, t_eval, ann, p)
  #rain:   rain intensity data (mm/s)
  #dry:    dry weather flow for each PT (L/s)
  #p_fix:  parameters to build the model (delayed mins,
  #        storage constant, catmentsize, PT, threshold value...)
  #u0:     initial value (L/s)
  #tspan:  time span for solving the ODEs
  #t_eval: time steps for evaluations
  #ann:    neural network
  #p:      parameters for neural network

  function catchment_(du, u, p, t)
    k = p_fix[6:10]
    theta = p_fix[11:15]
    area = p_fix[16:20]
    pt = p_fix[21:25]
    # B
    du[1] = (theta[1] * area[1] * rain(t) + pt[1]*dry(t)- u[1]
             + DiffEqFlux.restructure(ann, p[1:151])([pt[1]*dry(t), theta[1] * area[1] * rain(t)])[1])/k[1]
    # C
    du[2] = (theta[2] * area[2] * rain(t) + pt[2]*dry(t) - u[2]
             + DiffEqFlux.restructure(ann, p[1:151])([pt[2]*dry(t), theta[2] * area[2] * rain(t)])[1])/k[2]
    # D
    du[3] = (theta[3] * area[3]* rain(t) + pt[3]*dry(t) - u[3]
             + DiffEqFlux.restructure(ann, p[1:151])([pt[3]*dry(t), theta[3] * area[3] * rain(t)])[1])/k[3]
    # E
    du[4] = (theta[4] * area[4] * rain(t) + pt[4]*dry(t) - u[4]
             + DiffEqFlux.restructure(ann, p[1:151])([pt[4]*dry(t), theta[4] * area[4] * rain(t)])[1])/k[4]
    # F
    du[5] = (theta[5] * area[5] * rain(t) + pt[5]*dry(t) - u[5]
             + DiffEqFlux.restructure(ann, p[1:151])([pt[5]*dry(t), theta[5] * area[5] * rain(t)])[1])/k[5]
    end
  prob_ = ODEProblem(catchment_, u0, tspan, p2)
  function predict_adjoint()
    flow = diffeq_adjoint(p2, prob_, Tsit5(), u0 = u0_, saveat = t_eval)
    addOverflow(flow, p_fix[26:30])
  end
  return (predict_adjoint())
end


## add delay
function addDelay(sol, p, t_eval)
    num_cat = length(sol[1]) # number of catchment
    result = zeros(num_cat,length(t_eval))
    for i in 1:num_cat
        interp = LinearInterpolation(sol.t, sol[i, :])
        result[i,:] = interp(p[i]+1:p[i]+t_eval[end])
    end
    return (result)
end

## add overflow infractures influence
function addOverflow(flow, thres)
  outB = flow[1,:]
  outC = flow[2,:]
  outD = flow[3,:]
  outE = flow[4,:]
  outF = flow[5,:]
  outB = min.(outB, thres[1])
  outBC = min.(outB+outC, thres[2])
  outD = min.(outD, thres[3])
  outBCD = outBC+outD
  outE = min.(outE, 380)
  out_total = min.(outE+outF+outBCD, 75)
  return (out_total)
end
