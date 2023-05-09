module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff

export constrained_minimizer, constrained_optimal_value, complexgradient

function constrained_optimal_value(A, v, w=Nothing, P=(A.!=0))
    # @assert norm(v) ≈ 1 # removed since this will go in a tight loop
    Av = A*v
    m2 = sum(abs2.(v' .* P), dims=2)
    if w==Nothing
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda = 1im / norma * imag(v' * (Av ./ m2))
        w = v * lambda
    end
    z = w - Av
    optval = sum(abs2.(z) ./ m2)
    return optval
end

function constrained_minimizer(A, v, w=Nothing, P= (A.!=0))
    @assert norm(v) ≈ 1
    Av = A*v
    m2 = sum(abs2.(v' .* P), dims=2)
    if w==Nothing
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda = 1im / norma * imag(v' * (Av ./ m2))
        w = v * lambda
    end
    z = (w - Av) ./ m2
    E = z .* (v' .* P)
    return E
end

function complexgradient(f, cv)
   n = length(cv)
   gr = ForwardDiff.gradient(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
   return gr[1:n] + 1im * gr[n+1:end]
end

end