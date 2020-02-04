# batch train on the training data sets
function batchTrain(obs_data, Rain, Dry, ann, u_update, tspan, ts, n_itr, ps, fig)
    # obs_data: data sequence for training
    # ann: neural network that's used for prediction
    # u_update: updated u0 for the next batch training
    # tspan: the time span of each trainning batch
    # ts: time step for the result of the ode solver
    # n_itr: number of iteration for each batch data

    p1,re = Flux.destructure(ann)
    # p1: paramters of the NN
    # re: the structure of the NN

    num_batch = length(obs_data)
    for i in 1:num_batch
        u_update, p1 = russikon_nn(obs_data[i], Rain[i], Dry[i], u_update, p1, re, tspan, ts, n_itr, ps, fig)
    end
    # return the final trained parameters
    return u_update, p1
end

# training of one batch
function russikon_nn(carryonFlow, rain, dry, u0, p_ann, re, tspan, ts, n_itr, ps, fig)
  params = Flux.params(p_ann, u0)
  num_para = length(p_ann)

  # model
  function russikon_mix(du, u, p, t)
    k = ps[1:5]  # k: storage constant
    theta = ps[6:10]  # theta: runoff coefficient
    area = ps[11:15]  # area m^2
    areaRatio = ps[16:20]
    pt = ps[21:25]  # people eqivalent E

    neural = re(p[1:num_para])([rain(t), t])[1]
    # B
    du[1] = (theta[1] * area[1] * rain(t) + pt[1]*dry(t)- u[1] + areaRatio[1]*neural)/k[1]
    # C
    du[2] = (theta[2] * area[2] * rain(t) + pt[2]*dry(t) - u[2] + areaRatio[2]*neural)/k[2]
    # D
    du[3] = (theta[3] * area[3]* rain(t) + pt[3]*dry(t) - u[3] + areaRatio[3]*neural)/k[3]
    # E
    du[4] = (theta[4] * area[4] * rain(t) + pt[4]*dry(t) - u[4] + areaRatio[4]*neural)/k[4]
    # F
    du[5] = (theta[5] * area[5] * rain(t) + pt[5]*dry(t) - u[5] + areaRatio[5]*neural)/k[5]
  end

  prob_russikon_mix = ODEProblem(russikon_mix, u0, tspan, p_ann)

  # prediction
  function predict_n()
    #concrete_solve(prob_,Tsit5(),u_update, p1,saveat=ts; sensealg=TrackerAdjoint())
    concrete_solve(prob_russikon_mix,Tsit5(),u0, p_ann, saveat=ts; sensealg=TrackerAdjoint())
  end
  # add
  add() = addOverflow(predict_n())
  # loss function
  loss_russikon() = mean(abs2, carryonFlow - add())

  # training
  data = Iterators.repeated((), n_itr)    #number of iterations
  opt = ADAM(0.05)                        #algorithm
  cb = function()
    display(loss_russikon())
    # plot current prediction against obs data
    pred_nn = add()
    pl = plot(ts, carryonFlow, label = "obs")
    plot!(pl, ts, pred_nn, label = "pred")
    xlabel!("time_min")
    ylabel!("flow_L/s")
    display(pl)
  end

  if fig==1
    cb()
    Flux.train!(loss_russikon, params, data, opt) #, cb = cb())
    cb()
  else
    Flux.train!(loss_russikon, params, data, opt)
  end
  # update
  u_update = predict_n()[end]             #update u, note p1 is somehow automatically updated by Flux.train
  return u_update, p_ann

end

# evaluation of the model on test data sets
function evaluation(carryon, rain, dry, u_update, p1, re, tspan, ts, ps)
  params = Flux.params(p1, u_update)
  num_para = length(p1)
  # model
  #prob_toymodel = ODEProblem(toymodel, u_update, tspan, p1)
  function russikon_mix(du, u, p, t)
    k = ps[1:5]  # k: storage constant
    theta = ps[6:10]  # theta: runoff coefficient
    area = ps[11:15]  # area m^2
    areaRatio = ps[16:20]
    pt = ps[21:25]  # people eqivalent E

    neural = re(p[1:num_para])([rain(t), t])[1]
    # B
    du[1] = (theta[1] * area[1] * rain(t) + pt[1]*dry(t)- u[1] + areaRatio[1]*neural)/k[1]
    # C
    du[2] = (theta[2] * area[2] * rain(t) + pt[2]*dry(t) - u[2] + areaRatio[2]*neural)/k[2]
    # D
    du[3] = (theta[3] * area[3]* rain(t) + pt[3]*dry(t) - u[3] + areaRatio[3]*neural)/k[3]
    # E
    du[4] = (theta[4] * area[4] * rain(t) + pt[4]*dry(t) - u[4] + areaRatio[4]*neural)/k[4]
    # F
    du[5] = (theta[5] * area[5] * rain(t) + pt[5]*dry(t) - u[5] + areaRatio[5]*neural)/k[5]
  end
  prob_russikon_mix = ODEProblem(russikon_mix, u_update, tspan, p1)
  sol = concrete_solve(prob_russikon_mix,Tsit5(),u_update, p1, saveat=ts; sensealg=TrackerAdjoint())
  carryonflow = addOverflow(sol)
  pl = plot(ts, carryon, label = "obs")
  plot!(pl, ts, carryonflow, label = "pred")
  xlabel!("time_min")
  ylabel!("flow_L/s")
  display(plot(pl))
  mse = mean(abs2, carryonflow - carryon)
  return mse
end

# result of the model without using neural network
function russikon_original(carryon, rain, dry, u0, tspan, ts, ps)
  function russikon_model(du, u, p, t)
    k = ps[1:5]  # k: storage constant
    theta = ps[6:10]  # theta: runoff coefficient
    area = ps[11:15]  # area m^2
    areaRatio = ps[16:20]
    pt = ps[21:25]  # people eqivalent E
    infiltration = ps[26]
    # B
    du[1] = (theta[1] * area[1] * rain(t) + pt[1]*dry(t)- u[1] + areaRatio[1]*infiltration)/k[1]
    # C
    du[2] = (theta[2] * area[2] * rain(t) + pt[2]*dry(t) - u[2] + areaRatio[2]*infiltration)/k[2]
    # D
    du[3] = (theta[3] * area[3]* rain(t) + pt[3]*dry(t) - u[3] + areaRatio[3]*infiltration)/k[3]
    # E
    du[4] = (theta[4] * area[4] * rain(t) + pt[4]*dry(t) - u[4] + areaRatio[4]*infiltration)/k[4]
    # F
    du[5] = (theta[5] * area[5] * rain(t) + pt[5]*dry(t) - u[5] + areaRatio[5]*infiltration)/k[5]
  end
  prob_russikon_original = ODEProblem(russikon_model, u0, tspan, ps)
  sol = concrete_solve(prob_russikon_original,Tsit5(),u0, ps, saveat=ts; sensealg=TrackerAdjoint())
  carryonflow = addOverflow(sol)
  pl = plot(ts, carryon, label = "obs")
  plot!(pl, ts, carryonflow, label = "pred")
  xlabel!("time_min")
  ylabel!("flow_L/s")
  display(plot(pl))
  mse = mean(abs2, carryonflow-carryon)
  return mse
end

# add overflow infractures influence
function addOverflow(flow)
  outB = flow[1,:]
  outC = flow[2,:]
  outD = flow[3,:]
  outE = flow[4,:]
  outF = flow[5,:]
  outB = min.(outB, 200f0)
  outBC = min.(outB+outC, 250f0)
  outD = min.(outD, 90f0)
  outBCD = outBC+outD
  outE = min.(outE, 380f0)
  out_total = min.(outE+outF+outBCD, 75f0)
  return (out_total)
end

function batch_sep_rain(data, start, endt, num, span)
  # data: data set
  # num: number of batches
  # span: time span for each batch
  result = Dict()
  if num == 1
    index = findall(x->(x<=endt && x>= start), data.int_date)
    batch_data = data[index,:]
    batch_data.mins = batch_data.mins_hour.+1440*(batch_data.int_date.-start)
    interp = LinearInterpolation(batch_data.mins, batch_data.V3)
    interp_data(x) = convert(Float32, interp(x))
    return interp_data
  end

  for i in 1:num
    start_ = addDays(start, span*(i-1))
    end_ = addDays(endt, span*(i-1))
    index = findall(x->(x<=end_ && x>= start_), data.int_date)
    batch_data = data[index,:]
    batch_data.mins = batch_data.mins_hour.+1440*(batch_data.int_date.-start_)
    interp = LinearInterpolation(batch_data.mins, batch_data.V3)
    interp_data(x) = convert(Float32, interp(x))
    result[i] = interp_data
  end
  return result
end

function batch_sep_flow(data, start, endt, num, span, ts)
  # data: data set
  # num: number of batches
  # span: time span for each batch
  # ts: time steps for evaluation
  result = Dict()
  if num == 1
    index = findall(x->(x<=endt && x>= start), data.int_date)
    batch_data = data[index,:]
    batch_data.mins = batch_data.mins_hour.+1440*(batch_data.int_date.-start)
    interp = LinearInterpolation(batch_data.mins, batch_data.V3)
    interp_data = convert(Array{Float32, 1}, interp(ts))
    return interp_data
  end
  for i in 1:num
    start_ = addDays(start, span*(i-1))
    end_ = addDays(endt, span*(i-1))
    index = findall(x->(x<=end_ && x>= start_), data.int_date)
    batch_data = data[index,:]
    batch_data.mins = batch_data.mins_hour.+1440*(batch_data.int_date.-start_)
    interp = LinearInterpolation(batch_data.mins, batch_data.V3)
    interp_data = convert(Array{Float32, 1}, interp(ts))
    result[i] = interp_data
  end
  return result
end

# add days to dates
function addDays(date, diff)
  # change to Date form
  date_ = Date(string(date), "yyyymmdd")
  new_date = date_ + Dates.Day(diff)
  new_date_ = parse(Int64, Dates.format(new_date, "yyyymmdd"))
  return new_date_
end

# dry flow
function dry(data, num)
  result = Dict()
  interp = LinearInterpolation(data.mins, data.L_mins)
  dry(x) = convert(Float32, interp(x%1440)/60)
  if num==1
    return dry
  end
  for i in 1:num
    result[i] = dry
  end
  return result
end