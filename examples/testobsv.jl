using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, MosekTools
using SpecialFunctions
using ControlSystems



include("../src/WhiteBoxAnalysis.jl")




dim = 3; numMode = 3; dimOut = 1; horizon = 8;

#A = [[-0.1 -0.50 -0.4; 1 0.2 0.1; 0 -0.9 0.8], [-0.1 0.9 -0.4; 0.5 0.9 -0.8; -0.8 0.5 0.5], [0.5 0.1 0.4; 0.8 0.8 0.2; -0.2 -0.9 -0.5]]
#C = [rand(dimOut,dim) for i in 1:numMode]

A = [round.(Int,6*rand(dim,dim).-3) for i in 1:numMode]
C = [round.(Int,2*rand(dimOut,dim).-1) for i in 1:numMode]

#A =  [[-2 -2 3; 1 0 0; 1 2 1], [2 1 0; 0 3 -1; 2 -2 0], [1 2 1; -2 -3 -2; -2 1 -2]]
#C = [[1 0 0], [0 1 0], [0 0 1]]

#A = [[3 0 -3; 0 0 3; 1 2 -2], [-3 -1 2; -2 2 2; 2 -2 0], [-1 -2 -2; 2 2 1; -2 1 0]]
#C = [[-1 0 1], [-1 -1 -1], [1 0 0]]

#A = [[3 1 1; 2 0 2; -2 3 0], [2 -2 -2; -1 -2 0; 2 3 -2], [1 2 -1; 3 0 2; -2 3 0]]
#C = [[1 0 0], [-1 1 1], [1 0 -1]]

obsvindex = []
for i in 1:horizon
    obsv, = pathwise_obsv(A,C,i)
    push!(obsvindex,obsv)
    println(obsvindex)
end

println(obsvindex)

Umatrix = pathwise_obsv_subspace(A,C,5)
Umatrix = qr(Umatrix).Q * Matrix(I, size(Umatrix)...)