
function veroneselift(x,Xscale,d::Integer)
    X = collect((prod(y) for y in with_replacement_combinations(x, d)))
    X = X.*Xscale


    #=
    df = factorial(d)
    n = length(x)
    @polyvar y[1:n]
    scaling(m) = sqrt(df / prod(factorial, exponents(m)))
    Y = monomials(y, d)
    N = binomial(n+d-1, d)
    X = zeros(N)
    for (i,poly) in enumerate(Y)
        scaling_i = scaling(poly)
        power = exponents(poly)
        X[i] = prod(x[i]^power[i] for i in 1:n)*scaling_i
    end=#
    return X
end

function veroneseliftscale(n::Integer,d::Integer)
    power = collect(multiexponents(n, d))
    df = factorial(d)
    scaling(m) = sqrt(df / prod(factorial, m))
    Xscale = collect(scaling(poly) for poly in power)
    return Xscale
end

function kroneckerlift(x,d::Integer)
    xlift = copy(x)
    if d > 1
        for i in 1:d-1
            xlift = kron(xlift,x)
        end
    end
    return xlift
end

#=
function polyvector(Y,yvalue)
    Yvalue = []
    y = Y.vars
    for (i,poly) in enumerate(Y)
        append!(Yvalue, poly(y=>yvalue) )
    end
    return Yvalue
end
function kronecker2veronese(n::Integer,d::Integer)
    N = binomial(n+d-1, d)
    transform = spzeros(N,n^d)

    @polyvar y[1:n]
    Y = monomials(y, d)
    Ykron = kroneckerlift(y,d)
    df = factorial(d)
    scaling(m) = sqrt(df / prod(factorial, exponents(m)))
    for i in 1:N
        scaling_ij = scaling(Y[i])
        for j in 1:n^d
            if Y[i] == Ykron[j]
                transform[i,j] = scaling_ij
                break
            end
        end
    end
    return transform
end
=#

function kronecker2veronese(n::Integer,d::Integer)
    N = binomial(n+d-1, d)
    transform = spzeros(N,n^d)

    @polyvar y[1:n]
    Y = monomials(y, d)
    df = factorial(d)
    scaling(m) = sqrt(df / prod(factorial, exponents(m)))
    for i in 1:N
        scaling_ij = scaling(Y[i])
        power = exponents(Y[i])
        pos_i = zeros(n)
        cal_i = zeros(n)
        for k in 1:n
            if power[k] > 0
                if power[k] == 1
                    pos_i[k] = k
                else
                    pos_i[k] = (k-1)*(n^power[k]-n)/(n-1)+k
                end
                cal_i[k] = n^power[k]
            end
        end
        j = pos_i[1]
        for k in 2:n
            if pos_i[k] >=1
                if j >= 1
                    j = (j-1)*cal_i[k]+pos_i[k]
                else
                    j = pos_i[k]
                end
            end
        end
        j = Int(j)
        transform[i,j] = scaling_ij
    end
    return transform
end