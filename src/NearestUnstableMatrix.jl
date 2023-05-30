module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff

export constrained_minimizer, constrained_optimal_value, complexgradient, 
        #complexgradient_reverse, make_tape
        complexgradient_zygote

"""
    `optval = constrained_optimal_value(A, v, target, P=(A.!=0))`

Computes optval = min ||E||^2 s.t. (A+E)v = w and the constraint that the sparsity pattern of E is P (boolean matrix)

If target is a vector, w=target. Else target can be :LHP, :Disc, :0, and then w = v*λ, where λ
is chosen (outside the target or on its border) to minimize `constrained_optimal_value(A, v, vλ)`
"""
function constrained_optimal_value(A, v, target, P=(A.!=0))
    # @assert norm(v) ≈ 1 # removed since this will go in a tight loop

    # TODO: Reduce duplication!
    Av = A*v
    m2 = P * abs2.(v)
    if target === :LHP
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        if real(lambda0) >= 0
            lambda0 = lambda
        else
            lambda = 1im * imag(lambda0)
        end
        w = v * lambda
    elseif target === :Disc
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        al = abs(lambda0)
        if al >1
            lambda = lambda0
        else
            lambda = lambda0 / al
        end
        w = v * lambda
    elseif target === :0
        w = zeros(eltype(v), size(v))
    elseif isa(target, AbstractVector)
        w = target
    else
        error("Unknown target vector")
    end
    z = w - Av
    optval = sum(abs2.(z) ./ m2)
    return optval
end

"""
    `E = constrained_minimizer(A, v, target, P= (A.!=0))`

Computes the argmin corresponding to `constrained_optimal_value`
"""
function constrained_minimizer(A, v, target, P= (A.!=0))
    @assert norm(v) ≈ 1
    Av = A*v
    m2 = P * abs2.(v)
    if target === :LHP
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        if real(lambda0) >= 0
            lambda0 = lambda
        else
            lambda = 1im * imag(lambda0)
        end
        w = v * lambda
    elseif target === :Disc
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        al = abs(lambda0)
        if al >1
            lambda = lambda0
        else
            lambda = lambda0 / al
        end
        w = v * lambda
    elseif target === :0
        w = zeros(eltype(v), size(v))
    elseif isa(target, AbstractVector)
        w = target
    else
        error("Unknown target vector")
    end
    z = (w - Av) ./ m2
    E = z .* (v' .* P)
    return E
end

"""
    `g = complexgradient(f, cv)`

Computes the Euclidean gradient of a function f: C^n -> C^n (seen as C^n ≡ R^{2n}), using forward-mode AD
"""
function complexgradient(f, cv)
   n = length(cv)
   gr = ForwardDiff.gradient(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
   return gr[1:n] + 1im * gr[n+1:end]
end

using Zygote

"""
    `g = complexgradient_zygote(f, cv)`

As `complexgradient`, but uses reverse-mode AD in Zygote.
"""
function complexgradient_zygote(f, cv)
    return gradient(f, cv)
end

# Alternative implementation using ReverseDiff.
# Unfortunately it doesn't work as ReverseDiff does not like complex numbers even internally
#
# function make_tape(f, cv)
#     n = length(cv)
#     # actually it is enough to pass a vector of the right size, not the correct cv
#     f_tape = ReverseDiff.GradientTape(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)])
#     compiled_tape = compile(f_tape)
#     return compiled_tape
# end
# function complexgradient_reverse(cv, tape)
#     rv = [real(cv); imag(cv)]
#     results = similar(rv)
#     gradient!(results, tape, inputs)
# end

end