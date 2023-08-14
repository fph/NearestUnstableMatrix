module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff

import ChainRulesCore
using ChainRulesCore: ProjectTo, @not_implemented, @thunk


export constrained_minimizer, constrained_optimal_value, 
    realgradient, realgradient_zygote, #realgradient_reverse, make_tape,
    Disc, OutsideDisc, Hurwitz, Schur, Nonsingular, LeftHalfPlane
    
"""
    Custom type to wrap matrix multiplication, to work around an (apparent)
    bug with Zygote and constant sparse booleans.
"""
struct MatrixWrapper{T<:AbstractMatrix}
    P::T
end
(P::MatrixWrapper)(w::AbstractVector) = P.P*w
# rrule modified from https://discourse.julialang.org/t/zygote-product-with-a-constant-sparse-boolean/99510/3
function ChainRulesCore.rrule(P::MatrixWrapper, w::AbstractVecOrMat)
    project_w = ProjectTo(w)
    pullback(∂y) =  @not_implemented("A assumed constant"), project_w(P.P'*∂y)
    return P(w), pullback
end

abstract type Region end
struct Disc <: Region
    r::Float64
end
struct OutsideDisc <: Region
    r::Float64
end
const Schur = Disc(1.0)
const Nonsingular = OutsideDisc(0.0)
struct LeftHalfPlane <: Region
    re::Float64
end
const Hurwitz = LeftHalfPlane(0.0)

function project_outside(d::Disc, lambda)
    a = abs(lambda)
    return ifelse(a < d.r, lambda/a*d.r, lambda)
end
function project_outside(d::OutsideDisc, lambda)
    a = abs(lambda)
    return ifelse(a > d.r, lambda/a*d.r, lambda)
end
function project_outside(l::LeftHalfPlane, lambda)
    return ifelse(real(lambda)<l.re, lambda-real(lambda)+l.re, lambda)
end

"""
    `optval = constrained_optimal_value(A, v, target, P=(A.!=0))`

Computes optval = min ||E||^2 s.t. (A+E)v = w and the constraint that the sparsity pattern of E is P (boolean matrix)

If target is a vector, w=target. Else target can be :LHP, :Disc, :Nonsingular, and then w = v*λ, where λ
is chosen (outside the target or on its border) to minimize `constrained_optimal_value(A, v, vλ)`
"""
function constrained_optimal_value(A, v, target, P=(A.!=0); regularization=0.0)
    Av = A*v
    m2 = MatrixWrapper(P)(abs2.(v)) .+ regularization^2
    if isa(target, Region)
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        lambda = project_outside(target, lambda0)
        w = v * lambda
    elseif isa(target, AbstractVector)
        w = target
    else
        error("Unknown target specification")
    end
    z = w - Av
    optval = sum(abs2.(z) ./ m2)
    return optval
end

"""
Returns w such that constrained_optimal_value = norm(w)^2, for use in Levenberg-Marquardt-type algorithms
"""
function constrained_optimal_value_LM(A, v, target, P=(A.!=0); regularization=0.0)
    Av = A*v
    m2 = MatrixWrapper(P)(abs2.(v)) .+ regularization^2
    if isa(target, Region)
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        lambda = project_outside(target, lambda0)
        w = v * lambda
    elseif isa(target, AbstractVector)
        w = target
    else
        error("Unknown target specification")
    end
    z = w - Av
    fval = z ./ sqrt.(m2)
    return fval
end

"""
    `E = constrained_minimizer(A, v, target, P= (A.!=0))`

Computes the argmin corresponding to `constrained_optimal_value`
"""
function constrained_minimizer(A, v, target, P= (A.!=0); regularization=0.0)
    Av = A*v
    m2 = MatrixWrapper(P)(abs2.(v)) .+ regularization^2
    if isa(target, Region)
        norma = sqrt(sum(abs2.(v) ./ m2))
        lambda0 = (v' * (Av ./ m2)) / norma
        lambda = project_outside(target, lambda0)
        w = v * lambda
    elseif isa(target, AbstractVector)
        w = target
    else
        error("Unknown target specification")
    end
    z = (w - Av) ./ m2
    E = z .* (v' .* P)
    return E
end

"""
    `g = realgradient(f, cv)`

Computes the Euclidean gradient of a function f: C^n -> C^n (seen as C^n ≡ R^{2n}), using forward-mode AD
"""
function realgradient(f, cv)
   n = length(cv)
   gr = ForwardDiff.gradient(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
   return gr[1:n] + 1im * gr[n+1:end]
end
function realhessian(f, cv)
    n = length(cv)
    H = ForwardDiff.hessian(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
    return H
end

using Zygote

"""
    `g = realgradient_zygote(f, cv)`

As `realgradient`, but uses reverse-mode AD in Zygote.
"""
function realgradient_zygote(f, cv)
    return gradient(f, cv)
end
function realhessian_zygote(f, cv)
    # currently works only for dense matrices
    n = length(cv)
    H = Zygote.hessian(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
    return H
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
# function realgradient_reverse(cv, tape)
#     rv = [real(cv); imag(cv)]
#     results = similar(rv)
#     gradient!(results, tape, inputs)
# end

end