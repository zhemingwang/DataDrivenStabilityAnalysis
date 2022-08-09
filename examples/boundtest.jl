
using SpecialFunctions
using Plots
using LaTeXStrings

function betanonconvex(N,d,ep)

    temp = 0
    for i in d:N-1
        temp += binomial(big(i),d)*(1-ep)^(i-d)
    end
    return N*binomial(big(N),d)*(1-ep)^(N-d)/temp
    
end


function betaconvex(N,d,ep)
    return beta_inc(Float64(N-d+1),Float64(d),1-ep)[1]
end


N = 10000
d = 30
betaconvlist = []
betanonconvlist = []

for ep in 1e-4:1e-4:0.01
    betaconvxep = betaconvex(N,d,ep)
    betanonconvxep = betanonconvex(N,d,ep)
    push!(betaconvlist,betaconvxep)
    push!(betanonconvlist,betanonconvxep)
    println(betaconvxep)
    println(betanonconvxep)
end




gr(size = (600, 400))
fn = plot(1e-4:1e-4:0.01, 
Any[betaconvlist,betanonconvlist], 
label = ["Convex" "Nonconvex"],
line = [:solid :solid], 
xlabel=L"\epsilon",ylabel=L"\beta",
title = L"d=30", 
legend=:topright,
lw = 2,xtickfontsize=12,ytickfontsize=12,legendfontsize=12)

savefig(fn,"boundd=$d.png")