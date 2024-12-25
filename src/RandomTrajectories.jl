function generate_switched_linear_systems(numMode,dim)
    A = [randn(dim,dim) for i=1:numMode]
    for i = 1:numMode
        A[i] = A[i]/maximum(abs.(eigvals(A[i])))#opnorm(A[i])
    end
    return A
end


function generate_trajectories(horizon,A,C,numTraj)

numMode = size(A)[1]
dim = size(A[1])[1]
dimOut = size(C[1])[1]

y = zeros(dimOut*horizon,numTraj)

for i=1:numTraj
    z0 = randn(dim)
    x = z0/norm(z0)
    b = rand((1:numMode),horizon)
    for j = 1:horizon
        y[(j-1)*dimOut+1:j*dimOut,i] = C[b[j]]*x
        x = A[b[j]]*x
    end
end
return y
end

function generate_trajectories_with_noise(horizon, A, C, numTraj, noise_bound_w, noise_bound_v)
    numMode = size(A, 1)  
    dim = size(A[1], 1)   
    dimOut = size(C[1], 1)   

    y = zeros(dimOut * horizon, numTraj)

    for i = 1:numTraj
        z0 = randn(dim) 
        x = z0/norm(z0) 
        b = rand((1:numMode),horizon) 
        for j = 1:horizon
            w = 2 * noise_bound_w * rand(dim) .- noise_bound_w
            v = 2 * noise_bound_v * rand(dimOut) .- noise_bound_v   
            x = A[b[j]] * x + w
            y[(j-1)*dimOut+1:j*dimOut, i] = C[b[j]] * x + v  
        end
    end
    return y
end
