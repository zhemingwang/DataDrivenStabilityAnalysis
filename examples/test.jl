using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, MosekTools
using SpecialFunctions




include("../src/RandomTrajectories.jl")
include("../src/AlgebraicLift.jl")
include("../src/JSRCompute.jl")
include("../src/ProbabilisticCertificates.jl")


dim = 2; numMode = 2; dimOut = 2; horizon = 3;

numScen_budget = 5000

#A = [[-0.1 -0.50 -0.4; 1 0.2 0.1; 0 -0.9 0.8], [-0.1 0.9 -0.4; 0.5 0.9 -0.8; -0.8 0.5 0.5], [0.5 0.1 0.4; 0.8 0.8 0.2; -0.2 -0.9 -0.5]]
#C = [rand(dimOut,dim) for i in 1:numMode]

A = [rand(dim,dim) for i in 1:numMode]
#A = [[0.2851007425506096 0.9993496725475972; 0.21099560840381426 0.10654920270010781],[0.30862029423156057 0.8341408534411368; 0.32525344295679326 0.29158280836691897]]
C = [Matrix(1.0I, dim, dim) for i in 1:numMode]


jsrboundopen= white_box_jsr(A)
println("JSR open loop: $jsrboundopen")


traj_budget = generate_trajectories(horizon,A,C,numScen_budget)
hislen = 1
shift = horizon-hislen

state0_budget = traj_budget[1:hislen*dimOut,:]
state_budget = traj_budget[1+(horizon-hislen)*dimOut:end,:]


gamma_quad,boundcc,boundsa = probabilistc_stability_certificate(state0_budget,state_budget,numMode,shift,1,0.01)
gamma_sos,boundsos = probabilistc_stability_certificate(state0_budget,state_budget,numMode,shift,2,0.01)







