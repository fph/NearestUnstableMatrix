module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff

export constrained_minimizer, constrained_optimal_value, complexgradient

function constrained_optimal_value(A, v, w=Nothing, P=(A.!=0))
    @assert norm(v) ≈ 1
    n = size(A, 1)
    Av = A*v
    m = sqrt.(sum(abs2.(v' .* P), dims=2))
    if w==Nothing
        a = v ./ m
        b = (Av) ./ m
        lambda = 1im / norm(a) * imag(a'*b)
        w = v * lambda
    end
    z = w - Av
    optval = zero(eltype(v))
    for k = 1:n
        optval = optval + abs(z[k])^2 / m[k]^2
    end
    return optval
end

function constrained_minimizer(A, v, w=Nothing, P= (A.!=0))
    m = sqrt.(sum(abs2.(v' .* P), dims=2))
    if w==Nothing
        a = v ./ m
        b = (A*v) ./ m
        lambda = 1im / norm(a) * imag(a'*b)
        w = v * lambda
    end
    z = (w - A*v) ./ (m.^2)
    E = z .* (v' .* P)
    return E
end

function complexgradient(f, cv)
   n = length(cv)
   gr = ForwardDiff.gradient(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
   return gr[1:n] + 1im * gr[n+1:end]
end

end