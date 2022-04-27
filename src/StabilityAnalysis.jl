# data-driven jsr computation
module StabilityAnalysis

using Core: Matrix
using Random
using LinearAlgebra
using JuMP
using CSDP,MosekTools,SCS,ProxSDP
using HybridSystems
using SwitchOnSafety
using SpecialFunctions


# export generate_switched_linear_systems
# export white_box_jsr_compute
# export generate_trajectories
# export data_driven_lyapb_quad
# export parallel_scenario


include("JSRCompute.jl")
include("RandomTrajectories.jl")


end
