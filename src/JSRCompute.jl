function data_driven_lyapb_quad(state0,state;horizon=1,C=1e3,ub=1e2,lb=0,tol=1e-4,numIter=1e2,postprocess=false)
numTraj = size(state0)[2]
dim = size(state0)[1]
iter = 1
gammaU = ub
gammaL = lb
while gammaU-gammaL > tol && iter < numIter
    iter += 1
    gamma = (gammaU + gammaL)/2
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    #@variable(model, s>=0)
    @SDconstraint(model, P >= Matrix(I,dim,dim))
    @objective(model, Min, 0)
    for i in 1:numTraj
      @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
    end
    @SDconstraint(model, P <= C*Matrix(I,dim,dim))
    JuMP.optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
    #if value.(s) < 1e-10
      gammaU=gamma
    else
      gammaL=gamma
    end
end
gamma = gammaU

if postprocess == true
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    @SDconstraint(model, P >= Matrix(I,dim,dim))
    # @variable(model,t >= 0)
    # @constraint(model,t >= norm(P, 2))
    @objective(model, Min, sum(P[:].^2))
    for i in 1:numTraj
      @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
    end
    @SDconstraint(model, P <= C*Matrix(I,dim,dim))
    JuMP.optimize!(model)
    return gamma, value.(P)
else
    return gamma
end

end

function data_driven_lyapb_quad_unbounded(state0,state;horizon=1,ub=1e2,lb=0,tol=1e-4,numIter=1e2,postprocess=false)
    numTraj = size(state0)[2]
    dim = size(state0)[1]
    iter = 1
    gammaU = ub
    gammaL = lb
    while gammaU-gammaL > tol && iter < numIter
        iter += 1
        gamma = (gammaU + gammaL)/2
        solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, P[1:dim, 1:dim] in PSDCone())
        @variable(model, s>=0)
        @SDconstraint(model, P >= Matrix(I,dim,dim))
        @objective(model, Min, s)
        for i = 1:numTraj
          @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i]+s)
        end
        JuMP.optimize!(model)
        if value.(s) < 1e-6
            gammaU=gamma
          else
            gammaL=gamma
          end
    end
    gamma = gammaU
    
    if postprocess == true
        solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, P[1:dim, 1:dim] in PSDCone())
        @SDconstraint(model, P >= Matrix(I,dim,dim))
        # @variable(model,t >= 0)
        # @constraint(model,t >= norm(P, 2))
        @objective(model, Min, sum(P[:].^2))
        for i = 1:numTraj
          @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
        end
        JuMP.optimize!(model)
        return gamma, value.(P)
    else
        return gamma
    end
    
    end


function data_driven_lyapb_sos(state0,state,horizon,d,ga0,Xscale)

    n = size(state0,1)
    nlift = binomial(n+d-1, d)
    N = size(state0,2)

    statelift0 = zeros(nlift,N)
    statelift = zeros(nlift,N)
    for i in 1:N
        statelift0[:,i] = veroneselift(state0[:,i],Xscale,d)
        statelift[:,i] = veroneselift(state[:,i],Xscale,d)
    end

    ga,P =  data_driven_lyapb_quad(statelift0,statelift;horizon=horizon,C=1e5,ub=ga0^d,postprocess=true)
    #ga,P = data_driven_lyapb_quad_eigenmax(statelift0,statelift,ga0)

    return P,ga^(1/d)
end


function convergerate(state0,state,d,P,Xscale)
    N = size(state0,2)
    gamma = 0.0
    for i in 1:N
        xlift_i = veroneselift(state[:,i],Xscale,d)
        xlift0_i = veroneselift(state0[:,i],Xscale,d)
        gamma_i = sqrt(transpose(xlift_i)*P*xlift_i/(transpose(xlift0_i)*P*xlift0_i))
        if gamma_i > gamma
            gamma = copy(gamma_i)
        end
    end
    return gamma^(1/d)

end


