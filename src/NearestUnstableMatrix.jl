module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff

import ChainRulesCore
using ChainRulesCore: ProjectTo, @not_implemented, @thunk


export constrained_minimizer, constrained_optimal_value, 
    realgradient, realgradient_zygote, #realgradient_reverse, make_tape,
    Disc, OutsideDisc, Hurwitz, Schur, Nonsingular, LeftOf,
    nearest_eigenvector_outside, heuristic_zeros,
    MatrixWrapper, lambda_opt
    
"""
    Custom type to wrap matrix multiplication, to work around an (apparent)
    bug with Zygote and constant sparse booleans.
    Also, this ends up being faster than the custom sparse A*v in Zygote,
    because it does not compute the pullback wrt A, 
    so we switch to it for all our matrix products
"""
struct MatrixWrapper{T<:AbstractMatrix}
    P::T
end
(P::MatrixWrapper)(w::AbstractVecOrMat) = P.P * w
# rrule modified from https://discourse.julialang.org/t/zygote-product-with-a-constant-sparse-boolean/99510/3
function ChainRulesCore.rrule(P::MatrixWrapper, w::AbstractVecOrMat)
    project_w = ProjectTo(w)
    pullback(∂y) =  @not_implemented("A assumed constant"), project_w(P.P'*∂y)
    return P(w), pullback
end

abstract type Region end
struct Disc{r} <: Region
end
Disc(r) = Disc{r}()
struct OutsideDisc{r} <: Region
end
OutsideDisc(r) = OutsideDisc{r}()
const Schur = Disc{1.0}
const Nonsingular = OutsideDisc{0.0}
struct LeftOf{r} <: Region
end
LeftOf(r) = LeftOf{r}()
const Hurwitz = LeftOf{0.0}

function project_outside(d::Disc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a < r, lambda/a*r, lambda)
end
function project_outside(d::OutsideDisc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a > r, lambda/a*r, lambda)
end
function project_outside(l::LeftOf{r}, lambda) where r
    return ifelse(real(lambda)<r, lambda-real(lambda)+r, lambda)
end

inftozero(x) = ifelse(isinf(x), zero(x), x)

"""
    m2inv = compute_m2inv(v, regularization)
Computes the inverse of the weight vector, m2inv[i] = 1 / (||d_i||^2 + epsilon),
where d_i is the vector with d_i=1 iff A[i,j]≠0 and 0 otherwise.

Moreover, adjusts the vector such that 1/0=Inf is replaced by 0, to avoid NaNs in further computations.
"""
function compute_m2inv(P, v, regularization; warn=true)
    m2inv = 1 ./ (MatrixWrapper(P)(abs2.(v)) .+ regularization)
    if regularization == 0.
        m2inv = inftozero.(m2inv)
        if warn && any(v[m2inv .== 0]  .!= 0)
            @info "Unfeasible problem, you will probably need to set lambda = 0"
        end
    end 
    return m2inv
end

"""
    lambda_opt(Av, v, target, m2inv)

Computes the optimal projected eigenvalue lambda for a given problem
"""
lambda_opt(Av, v, target::Nonsingular, m2inv) = zero(eltype(v))
function lambda_opt(Av, v, target::Region, m2inv)
    denom = sum(abs2.(v) .* m2inv)
    numer = sum((conj.(v)) .* (Av .* m2inv))
    lambda = project_outside(target, numer / denom)
    return lambda
end

"""
    `optval = constrained_optimal_value(A, v, target, P=(A.!=0))`

Computes optval = min ||E||^2 s.t. (A+E)v = w and the constraint that the sparsity pattern of E is P (boolean matrix)

If target is a vector, w=target. Else target can be :LHP, :Disc, :Nonsingular, and then w = v*λ, where λ
is chosen (outside the target or on its border) to minimize `constrained_optimal_value(A, v, vλ)`
"""
function constrained_optimal_value(A, v, target, P=(A.!=0); regularization=0.0)
    Av = MatrixWrapper(A)(v)  # Av = A*v
    m2inv = compute_m2inv(P, v, regularization)
    lambda = lambda_opt(Av, v, target, m2inv)
    z = v * lambda - Av
    optval = sum(abs2.(z) .* m2inv)
    return optval
end

function constrained_optimal_value_Euclidean_gradient_zygote(A, v, target, P=(A.!=0); regularization=0.0)
    return first(realgradient_zygote(x -> constrained_optimal_value(A, x, target, P; regularization), v))
end

function constrained_optimal_value_Euclidean_gradient_analytic(A, v, target::Nonsingular, P=(A.!=0); regularization=0.0)
    m2inv = compute_m2inv(P, v, regularization)
    n = (A*v) .* m2inv
    grad = 2(A'*n - (P' * abs2.(n)) .* v)
    return grad
end


"""
Returns w such that constrained_optimal_value = norm(w)^2, for use in Levenberg-Marquardt-type algorithms
"""
function constrained_optimal_value_LM(A, v, target, P=(A.!=0); regularization=0.0)
    Av = MatrixWrapper(A)(v)  # Av = A*v
    m2inv = compute_m2inv(P, v, regularization)
    lambda = lambda_opt(Av, v, target, m2inv)
    w = v * lambda
    z = w - Av
    fval = z .* sqrt.(m2inv)
    return fval
end

"""
    `E, lambda = constrained_minimizer(A, v, target, P= (A.!=0))`

Computes the argmin corresponding to `constrained_optimal_value`
"""
function constrained_minimizer(A, v, target, P=(A.!=0); regularization=0.0)
    Av = MatrixWrapper(A)(v)  # Av = A*v
    m2inv = compute_m2inv(P, v, regularization)
    lambda = lambda_opt(Av, v, target, m2inv)
    z = v * lambda - Av
    t = z .* m2inv
    t[isnan.(t)] .= 0.
    E = t .* (v' .* P)  # the middle .* broadcasts column * row
    return E, lambda
end

"""
    Returns the augmented Lagrangian without the 1/reg*||y||^2 term, which is useless for minimization since it is constant
"""
function reduced_augmented_Lagrangian(A, v, y, target, P=(A.!=0); regularization=0.0)
    Av_mod = MatrixWrapper(A)(v) + regularization*y
    m2inv = compute_m2inv(P, v, regularization)
    lambda = lambda_opt(Av_mod, v, target, m2inv)
    w = v * lambda
    z = w - Av_mod
    optval = sum(abs2.(z) .* m2inv)
    return optval
end

function reduced_augmented_Lagrangian_minimizer(A, v, y, target, P=(A.!=0); regularization=0.0)
    Av_mod = MatrixWrapper(A)(v) + regularization*y
    m2inv = compute_m2inv(P, v, regularization)
    lambda = lambda_opt(Av_mod, v, target, m2inv)
    w = v * lambda
    z = w - Av_mod
    E = (z .* m2inv) .* (v' .* P)  # the middle .* broadcasts column * row
    return E, lambda
end

function reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target, P=(A.!=0); regularization=0.0)
    return first(realgradient_zygote(x -> reduced_augmented_Lagrangian(A, x, y, target, P; regularization), v))
end

function reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target::Nonsingular, P=(A.!=0); regularization=0.0)
    m2inv = compute_m2inv(P, v, regularization)
    n = (A*v + regularization * y) .* m2inv
    grad = 2(A'*n - (P' * abs2.(n)) .* v)
    return grad
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

using Manifolds, Manopt
"""
    nearest_eigenvector_outside(target, A, args...; optimizer=Manopt.trust_regions)

Computes v such that (A+E)v = v*lambda, lambda is _outside_ the target region, and ||E||_F is minimal
Additional keyword arguments are passed to the optimizer (for `debug`, `stopping_criterion`, etc.).
"""
function nearest_eigenvector_outside(target, A, x0; regularization=0.0, 
                                                    optimizer=Manopt.trust_regions, 
                                                    gradient=constrained_optimal_value_Euclidean_gradient_zygote,
                                                    kwargs...)
    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)

    f(M, v) = constrained_optimal_value(A, v, target; regularization)

    function g(M, v)
        gr = gradient(A, v, target; regularization)
        return project(M, v, gr)
    end

    x = optimizer(M, f, g, x0; kwargs...)
end

function penalty_method(target, A, x0; 
                        optimizer=Manopt.trust_regions, 
                        gradient=constrained_optimal_value_Euclidean_gradient_zygote,
                        iterations=30, 
                        starting_regularization=1.,
                        regularization_damping = 0.75, kwargs...)

    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)

    regularization = starting_regularization
    x0_warmstart = copy(x0)

    for k = 1:iterations
        E, lambda = constrained_minimizer(A, x0_warmstart, target; regularization)
        @show original_function_value = constrained_optimal_value(A, x0_warmstart, target)
        @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(A, x0_warmstart, target)[2]
        @show constraint_violation = norm((A+E)*x0_warmstart - x0_warmstart*lambda)
        @show regularization
        @show k

        f(M, v) = constrained_optimal_value(A, v, target; regularization)

        function g(M, v)
            gr = gradient(A, v, target; regularization)
            return project(M, v, gr)
        end
    
        x = optimizer(M, f, g, x0_warmstart; kwargs...)

        x0_warmstart .= x
        regularization = regularization * regularization_damping
    end
    @show original_function_value = constrained_optimal_value(A, x0_warmstart, target)
    @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(A, x0_warmstart, target)[2]
    return x0_warmstart
end


function augmented_Lagrangian_method(target, A, x0; optimizer=Manopt.trust_regions,
                                                    gradient=reduced_augmented_Lagrangian_Euclidean_gradient_zygote,
                                                    iterations=30,
                                                    starting_regularization=1., 
                                                    regularization_damping = 0.75,
                                                    kwargs...)
    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)

    y = zero(x0)  # this will be updated anyway
    regularization = starting_regularization
    x0_warmstart = copy(x0)
    for k = 1:iterations
        @show augmented_Lagrangian = reduced_augmented_Lagrangian(A, x0_warmstart, y, target; regularization) - regularization * norm(y)^2
        @show regularization
        @show k
        
        # We start with a dual gradient ascent step from x0 to get a plausible y0
        # dual gradient ascent.
        E, lambda = reduced_augmented_Lagrangian_minimizer(A, x0_warmstart, y, target; regularization)
        y .= y + (1/regularization) * ((A+E)*x0_warmstart - x0_warmstart*lambda)

        @show constraint_violation = norm((A+E)*x0_warmstart - x0_warmstart*lambda)
        @show original_function_value = constrained_optimal_value(A, x0_warmstart, target)
        @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(A, x0_warmstart, target)[2]
        
        f(M, v) = reduced_augmented_Lagrangian(A, v, y, target; regularization)

        function g_zygote(M, v)
            gr = gradient(A, v, y, target; regularization)
            return project(M, v, gr)
        end
    
        x = optimizer(M, f, g_zygote, x0_warmstart; kwargs...)

        x0_warmstart .= x
        regularization = regularization * regularization_damping
    end
    E, lambda = reduced_augmented_Lagrangian_minimizer(A, x0_warmstart, y, target; regularization)
    @show constraint_violation = norm((A+E)*x0_warmstart - x0_warmstart*lambda)
    @show original_function_value = constrained_optimal_value(A, x0_warmstart, target)
    @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(A, x0_warmstart, target)[2]
    return x0_warmstart
end

"""
    v = fix_unfeasibility!(A, v, target; P=(A.!=0))

Makes sure that the least-squares problem is solvable, i.e., that m2[i] == 0 => v[i] = 0.,
by inserting more zeros in v. This function is only needed if lambda != 0, otherwise the problem
is always solvable.
"""
function fix_unfeasibility!(A, v, target; P=(A.!=0))
    while(true)
        m2inv = compute_m2inv(P, v, 0.; warn=false)
        to_zero = (m2inv .== 0) .& (v .!= 0)
        if !any(to_zero)
            break
        end
        v[to_zero] .= 0.
    end
end

"""
    v = insert_zero_heuristic!(A, v, target; P=(A.!=0))

    Modifies v to insert zero values instead of small entries, hoping to reduce the objective function (but not guaranteed)
"""
function insert_zero_heuristic!(A, v, target; P=(A.!=0), force_lambda_zero=false)
    Av = MatrixWrapper(A)(v)
    if force_lambda_zero
        lambda = zero(eltype(v))
    else
        m2inv = compute_m2inv(P, v, 0.; warn=false)
        lambda = lambda_opt(Av, v, target, m2inv)
    end
    z = v * lambda - Av
    m2 = MatrixWrapper(P)(abs2.(v))
    lstsq_smallness = m2 + abs2.(z)
    _, i = findmin(x -> x==0 ? Inf : x,  lstsq_smallness)
    v[P[i,:] .!= 0.] .= 0.
    if !force_lambda_zero
        fix_unfeasibility!(A, v, target)
    end
end
"""
    v, fval = heuristic_zeros(A, v_, target; P=(A.!=0))

    Tries to replace with zeros some entries of v_ (those corresponding to small entries of m2), to get a lower 
    value fval for constrained_optimal_value.
"""
function heuristic_zeros(A, v_, target; P=(A.!=0), force_lambda_zero=false)
    v = copy(v_)
    bestval = constrained_optimal_value(A, v, target)
    bestvec = copy(v)
    for k = 1:length(v)
        insert_zero_heuristic!(A, v, target; P, force_lambda_zero)
        if iszero(v)
            break
        end
        curval = constrained_optimal_value(A, v, target)
        if curval < bestval
            bestval = curval
            bestvec .= v
        end
    end
    return bestvec, bestval
end


end # module
