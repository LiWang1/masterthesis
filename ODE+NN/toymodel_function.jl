function toy_model(n_itr, nn, tspan_end, init_u)
  # input

  Dry(x) = convert(Float32, sin(x)+5.0f0) # dry weather flow
  Rain(x) = convert(Float32, 5 *(sin(x)+abs(sin(x)))) #rain

  k1 = 10f0  #storage constant
  k2 = 15f0

  function trueODEfunc(du, u, p, t)
    du[1] = (Rain(t) + Dry(t) -u[1] + u[2] + 3.0f0+sin(t))/k1   # rain + dryflow - outflow + parasite water
    du[2] = (Rain(t) + Dry(t) -u[1] + u[2] + 3.0f0+sin(t))/k2
  end

  u0 = init_u
  tspan = (0.0f0, tspan_end)
  t = range(tspan[1], tspan[2], length = convert(Int64, floor(tspan_end))) #saveat
  prob = ODEProblem(trueODEfunc, u0, tspan)
  sol = concrete_solve(prob,Tsit5(), saveat = t)

  # ode + nn
  ann = nn
  p1,re = Flux.destructure(ann)
  ps = Flux.params(p1, u0)  # why here??
  num_para = length(p1)

  function dudt_(du, u, p, t)
    #x, y =
    neural = re(p[1:num_para])([Rain(t), t])[1]
    du[1]= (Rain(t) + Dry(t) -u[1] + u[2] + neural)/k1
    du[2] = (Rain(t) + Dry(t) -u[1] + u[2] + neural)/k2
  end

  prob_ = ODEProblem(dudt_, u0, tspan, p1)


  function predict_n()
    #concrete_solve(prob_,Tsit5(),u0,p1,saveat=t; sensealg=ForwardDiffSensitivity())
    concrete_solve(prob_,Tsit5(),u0, p1,saveat=t; sensealg=TrackerAdjoint())
  end

  penalty() = sum(norm, p1)
  loss_n() = sum(abs2, sol - predict_n()) + penalty()

  data = Iterators.repeated((), n_itr) #number of iterations
  opt = ADAM(0.05) #algorithm

  cb = function ()
    display(loss_n())
    #display(predict_m())
    # plot current prediction against data
    cur_pred = predict_n()
    pl = scatter(t, sol[1, :], label = "obs_outflow")
    scatter!(pl, t, sol[2, :], label = "obs_infiltration")
    scatter!(pl, t, cur_pred[1, :], label = "pred_outflow")
    scatter!(pl, t, cur_pred[2, :], label = "pred_infiltration")
    xlabel!("time_s")
    ylabel!("flow_L/s")
    display(plot(pl))
  end
  #cb()
  Flux.train!(loss_n, ps, data, opt)#, cb = cb)
end



# replace with nn
function toy_model_nn(n_itr, nn, tspan_end, init_u)
  # input
  k1 = 10f0  #storage constant
  k2 = 15f0

  function trueODEfunc(du, u, p, t)
    du[1] = (Rain(t) + Dry(t) -u[1] + u[2] + 3.0f0+sin(t))/k1   # rain + dryflow - outflow + parasite water
    du[2] = (Rain(t) + Dry(t) -u[1] + u[2] + 3.0f0+sin(t))/k2
  end

  u0 = init_u
  tspan = (0.0f0, tspan_end)
  t = range(tspan[1], tspan[2], length = convert(Int64, floor(tspan_end))) #saveat
  prob = ODEProblem(trueODEfunc, u0, tspan)
  sol = concrete_solve(prob,Tsit5(), saveat = t)

  # ode + nn
  ann = nn
  p1,re = Flux.destructure(ann)
  ps = Flux.params(p1, u0)  # why here??
  num_para = length(p1)

  function dudt_(du, u, p, t)
    #x, y =
    neural = re(p[1:num_para])([Rain(t), t])[1]
    du[1]= neural/k1
    du[2] = neural/k2
  end

  prob_ = ODEProblem(dudt_, u0, tspan, p1)


  function predict_n()
    #concrete_solve(prob_,Tsit5(),u0,p1,saveat=t; sensealg=ForwardDiffSensitivity())
    concrete_solve(prob_,Tsit5(),u0, p1,saveat=t; sensealg=TrackerAdjoint())
  end

  penalty() = sum(norm, p1)
  loss_n() = sum(abs2, sol - predict_n()) + penalty()

  data = Iterators.repeated((), n_itr) #number of iterations
  opt = ADAM(0.01) #algorithm

  cb = function ()
    display(loss_n())
    #display(predict_m())
    # plot current prediction against data
    cur_pred = predict_n()
    pl = scatter(t, sol[1, :], label = "obs_outflow")
    scatter!(pl, t, sol[2, :], label = "obs_infiltration")
    scatter!(pl, t, cur_pred[1, :], label = "pred_outflow")
    scatter!(pl, t, cur_pred[2, :], label = "pred_infiltration")
    xlabel!("time_s")
    ylabel!("flow_L/s")
    display(plot(pl))
  end
  cb()
  Flux.train!(loss_n, ps, data, opt, cb = cb)
end
