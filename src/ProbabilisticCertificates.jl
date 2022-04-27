function theta2delta(th,dim)
    return beta_inc((dim-1)/2,1/2,sin(th)^2)[1]
end
function delta2theta(delt,dim)
    sintheta = sqrt(beta_inc_inv((dim-1)/2,1/2,delt)[1])
    return asin(sintheta)
end

function epsilon2beta(ep,dim,numMode,N)
    return numMode*(1-theta2delta(delta2theta(ep,dim)/2,dim)/numMode)^N/theta2delta(delta2theta(ep,dim)/4,dim)
end


function beta2epsilon(beta,dim,numMode,N)
    ep_u = 1.0
    ep_l = 0.0
    while ep_u - ep_l > 1e-10
        ep = (ep_u+ep_l)/2
        beta_ep = epsilon2beta(ep,dim,numMode,N)
        if beta_ep > beta
            ep_l = ep
        else
            ep_u = ep
        end
    end
    return ep_u
end

function beta_tailed_custom(ep,d,N)
    beta_violation = 0
    if ep <= 1/d
        for k = 1:d
            beta_violationtemp = 0
            for j = 0:d-k
                beta_violationtemp += binomial(k-1+j,j)*(1-k*ep)^N
            end
            beta_violation += (-1)^(k-1)*beta_violationtemp
        end
    end
    beta_violation = 1 - beta_violation
    beta_conservative = d*(1-ep)^N
    if beta_violation > beta_conservative
        beta_violation = beta_conservative
    end
    return beta_violation
end



function beta_tailed_inv_custom(beta,d,N)
    ep_u = 1.0/d
    ep_l = 0.0

    while ep_u - ep_l > 1e-10
        ep = (ep_u+ep_l)/2
        s = beta_tailed_custom(ep,d,N)
        if s < beta
            ep_u = ep
        else
            ep_l =  ep
        end
    end

    return (ep_u+ep_l)/2
end



function probabilistc_stability_certificate(state0,state,numMode,horizon,d,beta=0.01)
    dim = size(state0,1)
    numTraj = size(state0,2)

    Xscale = veroneseliftscale(dim,d)

    dimlift = binomial(dim+d-1, d)
    P0 = Matrix(1.0I, dimlift, dimlift)
    ga0 = convergerate(state0,state,d,P0,Xscale)
    ga0 = ga0^(1/horizon)
    
    P,gamma = data_driven_lyapb_sos(state0,state,horizon,d,ga0,Xscale)
    traj_norm = norm_max(state0,state)

    if d == 1
        boundcc = data_driven_jsr_chanceconstraint(gamma,beta,P;numTraj,numMode,horizon)
        println("Chance-cosntrained bound: $boundcc")
        boundsa = data_driven_jsr_sensitivityanalysis_quad(gamma,beta,P,traj_norm;numTraj,numMode,horizon)
        println("Sentivity-analysis bound (quadratic): $boundsa")

        return gamma,boundcc, boundsa
    else
        boundsos = data_driven_jsr_sensitivityanalysis_sos(gamma,beta,P,traj_norm;dim,numTraj,numMode,d,horizon)
        println("Sentivity-analysis bound (sos): $boundsos")

        return gamma,boundsos
    end

end




function data_driven_jsr_chanceconstraint(gamma,beta,P;numTraj,numMode,horizon=1)
    dim = size(P)[1]
    kappaP = sqrt(det(P)/eigmin(P)^dim)
    d = (dim+1)*dim/2
    ep = 1- beta_inc_inv(Float64(numTraj-d+1),Float64(d),beta)[1]
    epM = ep*numMode^horizon
    if epM < 1
        ep_kappa = epM*kappaP
        if ep_kappa <1
            delt = sqrt(1-beta_inc_inv((dim-1)/2.0,0.5,ep_kappa)[1])
        else
            delt = sqrt(1-beta_inc_inv((dim-1)/2.0,0.5,1-(1-ep*numMode^horizon)*sqrt(det(P)/eigmax(P)^dim))[1])
        end
        jsr_bound = gamma/delt^(1/horizon)
        return jsr_bound
    else
        println("More samples are needed!")
        return 1e6
    end
    
end
    
function data_driven_jsr_sensitivityanalysis_quad_tuning(gamma,beta,P,traj_norm,betanorm;numTraj,numMode,horizon=1)
    dim = size(P)[1]
    kappaP = sqrt(eigmax(P)/eigmin(P))
    dv = Int64((dim+1)*dim/2)
    ep = numMode^horizon*beta_tailed_inv_custom(beta,dv,numTraj)/2 #numMode^horizon*(1-((beta)/d)^(1/numTraj))/2 #
    if ep < 0.5
        delt = sqrt(1-beta_inc_inv((dim-1)/2.0,0.5,2*ep)[1])
        Delt = sqrt(2-2*delt)
        normbound = data_driven_jsr_norm(traj_norm,betanorm;dim=dim,numMode=numMode,numTraj=numTraj,horizon=horizon)
        jsr_bound = (gamma^horizon+(gamma^horizon+normbound^horizon)*Delt*kappaP)^(1/horizon)
        return jsr_bound
    else
        println("More samples are needed!")
        return 1e6
    end

end

    
function data_driven_jsr_sensitivityanalysis_quad(gamma,beta,P,traj_norm;numTraj,numMode,horizon=1)
    jsr_bound = 1e6
    for beta1 = 0.1*beta:0.1*beta:0.9*beta
        jsr_boundtemp = data_driven_jsr_sensitivityanalysis_quad_tuning(gamma,beta1,P,traj_norm,beta-beta1;numTraj,numMode,horizon)
        if jsr_boundtemp < jsr_bound
            jsr_bound = jsr_boundtemp
        end
    end
    return jsr_bound
end
    
    
    
function data_driven_jsr_norm(traj_norm,beta;dim,numMode,numTraj,horizon=1)

    epnorm = numMode^horizon*(1-(beta)^(1/numTraj))/2
    if epnorm < 0.5 
        deltnorm = sqrt(1-beta_inc_inv((dim-1)/2.0,0.5,2*epnorm)[1])
        normbound = (traj_norm/deltnorm)^(1/horizon)
        return normbound
    else
        println("More samples are needed!")
        return 1e6
    end
end
    
function norm_max(state0,state)
    traj_norm = 0.0
    numTraj = size(state0)[2]

    for i = 1:numTraj
        traj_norm_i = norm(state[:,i])/norm(state0[:,i])
        if traj_norm_i > traj_norm
            traj_norm = traj_norm_i
        end
    end

    return traj_norm

end
    

function data_driven_jsr_sensitivityanalysis_sos_tuning(gamma,beta,P,traj_norm,betanorm;dim,numTraj,numMode,d,horizon=1)
    dimlift = size(P)[1]
    kappaP = sqrt(eigmax(P)/eigmin(P))
    dv = Int64((dimlift+1)*dimlift/2)
    ep = numMode^horizon*beta_tailed_inv_custom(beta,dv,numTraj)/2 #numMode^horizon*(1-((beta)/d)^(1/numTraj))/2 #
    if ep < 0.5
        phid = 0.0
        for i in 1:d
            phid += (2-2*cos(delta2theta(ep,dim)))^(i/2)*binomial(d,i)
        end
        normbound = data_driven_jsr_norm(traj_norm,betanorm;dim=dim,numMode=numMode,numTraj=numTraj,horizon=horizon)
        bound = (gamma^d*(1+sqrt(kappaP)*phid)+normbound^d*sqrt(kappaP)*phid)^(1/d)
        return bound
    else
        println("More samples are needed!")
        return 1e6
    end
    
end
function data_driven_jsr_sensitivityanalysis_sos(gamma,beta,P,traj_norm;dim,numTraj,numMode,d,horizon=1)
    jsr_bound = 1e6
    for beta1 = 0.1*beta:0.1*beta:0.9*beta
        jsr_boundtemp = data_driven_jsr_sensitivityanalysis_sos_tuning(gamma,beta1,P,traj_norm,beta-beta1;dim,numTraj,numMode,d,horizon)
        if jsr_boundtemp < jsr_bound
            jsr_bound = jsr_boundtemp
        end
    end
    return jsr_bound
end
            