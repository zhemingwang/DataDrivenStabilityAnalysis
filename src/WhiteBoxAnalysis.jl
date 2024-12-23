function white_box_jsr(A,d=2)
    s = discreteswitchedsystem(A)
    optimizer_constructor = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    soslyapb(s, d, optimizer_constructor=optimizer_constructor, tol=1e-4, verbose=1)
    seq = sosbuildsequence(s, d, p_0=:Primal)
    psw = findsmp(seq)
    return psw.growthrate
end


function path_generate(numMode,horizon)
    pathrun = []
    for i in 1:numMode
        push!(pathrun,[i])
    end
    if horizon > 1
        for t in 1:horizon-1
            pathrunt = []
            for p in pathrun
                for i in 1:numMode
                    push!(pathrunt,vcat(p,i))
                end
            end
            pathrun = copy(pathrunt)
        end
    end
    return pathrun
end


function path_dependent_obsv(A,C,horizon)
    numMode = size(A,1)
    n = size(A[1],1)
    p = size(C[1],1)

    obsv = 0
    obsvsq = []

    pathlist = path_generate(numMode,horizon)
    for path in pathlist
        obsvmat = zeros(p*horizon,n)
        Arun = Matrix(I,n,n)
        for t in 1:horizon
            obsvmat[p*(t-1)+1:t*p,:] = C[path[t]]*Arun
            Arun = A[path[t]]*Arun
        end
        if rank(obsvmat) == n
            obsv = 1 
            obsvsq = path
            break
        end
    end

    return obsv,obsvsq
end

function pathwise_obsv(A,C,horizon)
    numMode = size(A,1)
    n = size(A[1],1)
    p = size(C[1],1)

    obsv = 1
    obsvsq = []

    pathlist = path_generate(numMode,horizon)
    for path in pathlist
        obsvmat = zeros(p*horizon,n)
        Arun = Matrix(I,n,n)
        for t in 1:horizon
            obsvmat[p*(t-1)+1:t*p,:] = C[path[t]]*Arun
            Arun = A[path[t]]*Arun
        end
        if rank(obsvmat) < n
            obsv = 0 
            obsvsq = path
            break
        end
    end
    return obsv,obsvsq
end





function white_box_lyap_quad(A,C,obsindex,horizon,tol=1e-4)
  normAmax = 0
  dim = size(A[1],1)
  for Ai in A
      normAi = opnorm(Ai)
      if normAi>normAmax
          normAmax = normAi
      end
  end
  ub = normAmax
  lb = 0
  P = Matrix(I,dim,dim)
  while ub-lb > tol
      gamma = (ub + lb)/2
      solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
      model = Model(solver)
      @variable(model, Q[1:dim, 1:dim] in PSDCone())
      @variable(model, Y[1:dimIn,1:dim])
      @variable(model, s>=0)
      @constraint(model, Q >= Matrix(I,dim,dim),PSDCone())
      @objective(model, Min, s)
      for Ai in A
          @constraint(model, [gamma^2*Q+s*Matrix(I,dim,dim) Q*Ai'+(B*Y)';Ai*Q+B*Y Q+s*Matrix(I,dim,dim)] >= 0,PSDCone())
      end
      JuMP.optimize!(model)
      if value.(s) < 1e-10
          ub = gamma
          P = inv(value.(Q))
      else
          lb = gamma
      end
  end
  #=solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
  model = Model(solver)
  @variable(model, Q[1:dim, 1:dim] in PSDCone())
  @variable(model, Y[1:dimIn,1:dim])
  @variable(model, t)
  @SDconstraint(model, Q >= Matrix(I,dim,dim))
  @constraint(model, [t; 1; vec(Q)] in MOI.LogDetConeSquare(dim))
  @objective(model, Max, t)
  for Ai in A
      @SDconstraint(model, [ub^2*Q Q*Ai'+(B*Y)';Ai*Q+B*Y Q] >= 0)
  end
  JuMP.optimize!(model)
  P = inv(value.(Q))
  K = Y*P=#
  return ub, K, P

end
