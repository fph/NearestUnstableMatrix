module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff
using SparseArrays
using DataFrames

using ChainRulesCore
using ChainRulesCore: ProjectTo, @not_implemented, @thunk
import Manifolds: project

export constrained_minimizer, constrained_optimal_value, constrained_AplusE,
    realgradient, realgradient_zygote,
    InsideDisc, OutsideDisc, NonHurwitz, NonSchur,  RightOf, Singular,
    precompute, ComplexSparsePerturbation, GeneralPerturbation,
    nearest_unstable, heuristic_zeros,
    ConstantMatrixProduct, lambda_opt,
    project, toeplitz_perturbation, grcar
    
"""
    Custom type to wrap matrix multiplication with a constant matrix A, 
    to work around an (apparent) bug with Zygote and constant sparse booleans.
    Also, this ends up being faster than the custom sparse A*v in Zygote,
    because it does not compute the pullback wrt A, 
    so we switch to it for all our matrix products.
"""
struct ConstantMatrixProduct{T<:AbstractMatrix}
    P::T
end
(P::ConstantMatrixProduct)(w::AbstractVecOrMat) = P.P * w
# rrule modified from https://discourse.julialang.org/t/zygote-product-with-a-constant-sparse-boolean/99510/3
function ChainRulesCore.rrule(P::ConstantMatrixProduct, w::AbstractVecOrMat)
    project_w = ProjectTo(w)
    pullback(∂y) =  @not_implemented("A assumed constant"), project_w(P.P'*∂y)
    return P(w), pullback
end

abstract type Region end
struct InsideDisc{r} <: Region
end
InsideDisc(r) = InsideDisc{r}()
struct OutsideDisc{r} <: Region
end
OutsideDisc(r) = OutsideDisc{r}()
const NonSchur = OutsideDisc{1.0}
const Singular = OutsideDisc{0.0}
struct RightOf{r} <: Region
end
RightOf(r) = RightOf{r}()
const NonHurwitz = RightOf{0.0}

function project(d::OutsideDisc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a < r, lambda/a*r, lambda)
end
function project(d::InsideDisc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a > r, lambda/a*r, lambda)
end
function project(l::RightOf{r}, lambda) where r
    return ifelse(real(lambda)<r, lambda-real(lambda)+r, lambda)
end

abstract type PerturbationStructure end

"""
    Type for a complex sparse perturbation with sparsity structure P. P should be a BitMatrix or something compatible.
"""
struct ComplexSparsePerturbation{T} <: PerturbationStructure where T <: AbstractMatrix{Bool}
    P::T
end

"""
    Type for a general perturbation E = ∑ Eᵢ ωᵢ. EE is a vector of matrices. Probably this is not going to be too performant.
"""
struct GeneralPerturbation{T} <: PerturbationStructure where T<:AbstractVector
    EE::T
end

function GeneralPerturbation(pert::ComplexSparsePerturbation)
    m, n = size(pert.P)
    (ii, jj, _) = findnz(sparse(pert.P))
    EE = collect(sparse([i], [j], [1.], m, n) for (i, j) in zip(ii, jj))
    P2 = GeneralPerturbation(EE)
end

"""
    toeplitz_perturbation(n)
    toeplitz_perturbation(A)

    Return a GeneralPerturbation that allows (complex) Toeplitz perturbations of a n×n matrix.
"""
function toeplitz_perturbation(n::Number, indices=-(n-1):(n-1))
    EE = collect(1/sqrt(n-abs(k)) * diagm(k => ones(n-abs(k))) for k in indices)
    return GeneralPerturbation(EE)
end
toeplitz_perturbation(A::Matrix) = toeplitz_perturbation(size(A,1))
toeplitz_perturbation(A::Matrix, indices) = toeplitz_perturbation(size(A,1), indices)

"""
    grcar(n)

Compute the nxn Grcar matrix
"""
function grcar(n)
    return diagm(-1=>-ones(n-1), 0=>ones(n), 1=>ones(n-1), 2=>ones(n-2), 3=>ones(n-3))
end

function compute_MvMvt(pert::ComplexSparsePerturbation, v; regularization=0.0)
    d2 = ConstantMatrixProduct(pert.P)(abs2.(v))
    return Diagonal(d2)
end

function compute_Mv(pert::GeneralPerturbation, v)
    return reduce(hcat, E*v for E in pert.EE)
end


inftozero(x) = ifelse(isinf(x), zero(x), x)
"""
    pc = precompute(pert, v, regularization)

Computes "something related" to the inverse of the weighting matrix M_v.

* For ComplexSparsePerturbation, it is the inverse of the weight vector, 
m2inv[i] = 1 / (||d_i||^2 + epsilon), where d_i is the vector with d_i=1 iff A[i,j]≠0 and 0 otherwise.
Moreover, the method adjusts the vector such that 1/0=Inf is replaced by 0, to avoid NaNs in further computations.

* For GeneralPerturbation, it is QR(M_v)
"""
function precompute(pert::ComplexSparsePerturbation, v, regularization; warn=true)
    m2inv = 1 ./ (ConstantMatrixProduct(pert.P)(abs2.(v)) .+ regularization)
    if regularization == 0.
        m2inv = inftozero.(m2inv)
        if warn && any(v[m2inv .== 0]  .!= 0)
            @info "Unfeasible problem, you will probably need to set lambda = 0"
        end
    end
    return m2inv
end
function precompute(pert::GeneralPerturbation, v, regularization; warn=true)
    Mv = compute_Mv(pert, v)
    if iszero(regularization)
        Mv_reg = Mv
    else
        Mv_reg = hcat(Mv, sqrt(regularization)*I)
    end
    return qr(Mv_reg')
end

"""
    Computes the value of the quadratic form xᴴ(MᵥMᵥᴴ+λI)⁻¹x, 
"""
function inverse_quadratic_form(pc::LinearAlgebra.QRCompactWY, x)
    a = pc.R' \ x
    return sum(abs2, a)
end
function inverse_quadratic_form(pc::Vector, x)
    return sum(abs2.(x) .* pc)
end

"""
    Computes the value of the bilinear form yᴴ(MᵥMᵥᴴ+λI)⁻¹x, 
"""
function inverse_bilinear_form(y, pc::LinearAlgebra.QRCompactWY, x)
    a = pc.R' \ x
    b = pc.R' \ y
    return dot(b, a)
end
function inverse_bilinear_form(y, pc::Vector, x)
    sum((conj.(y)) .* (x .* pc))
end

"""
Computes (MᵥMᵥᴴ+λI)⁻¹x
"""
function inverse_quadratic_form_matvec(pc::LinearAlgebra.QRCompactWY, z)
    return pc.R \ (pc.R' \ z)
end
function inverse_quadratic_form_matvec(pc::Vector, z)
    return z .* pc
end

"""
    Computes the smallest E s.t. Ev = z  
"""
function optimal_E(pert::GeneralPerturbation, v, z, pc)
    omega = (pc.Q*(pc.R' \ z))[1:length(pert.EE)]
    E = reduce(+, E*m for (E,m) in zip(pert.EE, omega))
    return E
end
function optimal_E(pert::ComplexSparsePerturbation, v, z, pc)
    t = z .* pc
    E = t .* (v' .* pert.P)  # the middle .* broadcasts column * row    
    return E
end

"""
    Return ∇ zᴴ(MᵥMᵥᴴ)⁻¹z,
    for a vector z with dz/dv = (A-λ I)  --- typically, z = (A-λI)v + εy), to account for the extended Lagrangian formulation
"""
function analytic_gradient(pert::GeneralPerturbation, A, v, z, lambda, pc)
    invRtz = pc.R' \ z
    omega = (pc.Q*invRtz)[1:length(pert.EE)]
    nv = pc.R \ invRtz
    grad = 2(A'*nv - nv*lambda' - sum(conj(m)*(E'*nv) for (E,m) in zip(pert.EE, omega)))
    return grad
end
function analytic_gradient(pert::ComplexSparsePerturbation, A, v, z, lambda, pc)
    nv = z .* pc
    grad = 2(A'*nv - nv*lambda' - (pert.P' * abs2.(nv)) .* v)
    return grad
end

"""
    lambda_opt(target, pert, Av, v, pc)

Computes the optimal projected eigenvalue lambda for a given problem
"""
function lambda_opt(target::Region, pert, Av, v, pc)
    denom = inverse_quadratic_form(pc, v)
    numer = inverse_bilinear_form(v, pc, Av)
    lambda = project(target, numer / denom)
    return lambda
end
lambda_opt(target::Singular, pert, Av, v, pc) = zero(eltype(v))

"""
    Computes lambda_opt from R⁻ᴴv
"""
function lambda_opt_from_invRtv(target, pert::GeneralPerturbation, Av, invRtv, pc)
    b = pc.R' \ Av
    a = invRtv
    lambda = project(target, dot(a,b) / dot(a,a))
    return lambda
end
lambda_opt_from_invRtv(target::Singular, pert::GeneralPerturbation, Av, invRtv, pc) = zero(eltype(v))

"""
    `E, lambda = constrained_minimizer(target, pert A, v; regularization=0.0)`

Computes the argmin corresponding to `constrained_optimal_value`
"""
function constrained_minimizer(target, pert, A, v; regularization=0.0)
    Av = ConstantMatrixProduct(A)(v)  # Av = A*v
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av, v, pc)
    z = v * lambda - Av
    E = optimal_E(pert, v, z, pc)
    return E, lambda
end

"""
    `AplusE, lambda = constrained_AplusE(target, pert, alpha, v; regularization=0.0)`

Computes the value of A+E
"""
function constrained_AplusE(target, pert::ComplexSparsePerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = A*v
    else
        Av = A*v + regularization * y
    end
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av, v, pc)
    nv = pc .* (lambda*v - Av)
    t1 = (v*lambda) .* pc
    vP = v' .* pert.P  # matrix with the nonzero structure of A but elements conj(v_j)
    E1 = t1 .* vP # broadcasting
    s2 = pert.P * abs2.(v)
    projA = A - ((Av) ./ s2) .* vP # this projects A on Im(pc.V)^⟂
    # we subtract this projection before the next addition in the hope of getting
    # exact zeros where needed, and avoid losing accuracy.
    AplusE = projA + E1 + (Diagonal(regularization ./ (s2 .+ regularization) ./ s2) *Av) .* vP
    return AplusE, lambda, nv
end

function constrained_gradient_alternative(target, pert, A, v, y=nothing; regularization=0.0)
    AplusE, lambda, nv = constrained_AplusE(target, pert, A, v, y; regularization)
    return 2(nv*lambda' - AplusE'*nv)
end


"""
    `optval = constrained_optimal_value(target, pert, A, v)`

Computes optval = min ||E||^2 s.t. (A+E)v = v*λ and the constraint that the sparsity pattern of E is P (boolean matrix)

λ is chosen (inside the target region or on its border) to minimize `constrained_optimal_value(A, v, λ)`
"""
function constrained_optimal_value(target, pert, A::AbstractMatrix, v; regularization=0.0)
    Av = ConstantMatrixProduct(A)(v)  # Av = A*v
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av, v, pc)
    z = v * lambda - Av
    optval = inverse_quadratic_form(pc, z)
    return optval
end

function constrained_optimal_value(target, pert::GeneralPerturbation, alpha::AbstractVector, v; regularization=0.0)
    Mv = compute_Mv(pert, v) # TODO: optimize out, we already compute it in `precompute`
    Av = Mv * alpha
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    invRtv = pc.R' \ v
    lambda = lambda_opt_from_invRtv(target, pert, Av, invRtv, pc)
    alpha_ext = regularization==0.0 ? alpha : [alpha;zeros(length(v))] 
    mv = pc.Q'*alpha_ext
    aux = pc.Q*(invRtv * lambda - mv[1:length(v)])
    return sum(abs2, aux[1:length(pert.EE)])
end


function constrained_optimal_value_from_z(target, pert, A, z, lambda; regularization=0.0)
    v = (A-lambda*I) \ z
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    optval = inverse_quadratic_form(pc, z)
    return optval
end

function constrained_optimal_value_Euclidean_gradient_zygote(target, pert, A, v; regularization=0.0)
    return first(realgradient_zygote(x -> constrained_optimal_value(target, pert, A, x; regularization), v))
end

function constrained_optimal_value_Euclidean_gradient_analytic(target, pert, A, v; regularization=0.0)
    Av = ConstantMatrixProduct(A)(v)  # Av = A*v
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av, v, pc)
    z = Av - v*lambda
    return analytic_gradient(pert, A, v, z, lambda, pc)
end


# """
# Returns w such that constrained_optimal_value = norm(w)^2, for use in Levenberg-Marquardt-type algorithms
# """
# function constrained_optimal_value_LM(target, pert, A, v; regularization=0.0)
#     Av = MatrixWrapper(A)(v)  # Av = A*v
#     pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
#     m2inv = pc
#     lambda = lambda_opt(target, pert, Av, v, m2inv)
#     w = v * lambda
#     z = w - Av
#     fval = z .* sqrt.(m2inv)
#     return fval
# end

"""
    Returns the augmented Lagrangian without the 1/reg*||y||^2 term, which is useless for minimization since it is constant
"""
function reduced_augmented_Lagrangian(target, pert, A, v, y; regularization=0.0)
    Av_mod = ConstantMatrixProduct(A)(v) + regularization*y
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av_mod, v, pc)
    z = v * lambda - Av_mod
    optval = inverse_quadratic_form(pc, z)
    return optval
end

function reduced_augmented_Lagrangian_minimizer(target, pert, A, v, y; regularization=0.0)
    Av_mod = ConstantMatrixProduct(A)(v) + regularization*y
    pc =  precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av_mod, v, pc)
    z = v * lambda - Av_mod
    E = optimal_E(pert, v, z, pc)
    return E, lambda
end

function reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, pert, A, v, y; regularization=0.0)
    return first(realgradient_zygote(x -> reduced_augmented_Lagrangian(target, pert, A, x, y; regularization), v))
end

function reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, pert, A, v, y; regularization=0.0)
    Av_mod = ConstantMatrixProduct(A)(v) + regularization*y
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    lambda = lambda_opt(target, pert, Av_mod, v, pc)
    z = Av_mod - v*lambda
    return analytic_gradient(pert, A, v, z, lambda, pc)
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

using Manifolds, Manopt
"""
    nearest_unstable(target, pert, A, args...; optimizer=Manopt.trust_regions)

Computes v such that (A+E)v = v*lambda, lambda is inside the target region, and ||E||_F is minimal
Additional keyword arguments are passed to the optimizer (for `debug`, `stopping_criterion`, etc.).
"""
function nearest_unstable(target, pert, A, x0; regularization=0.0, 
                                                    optimizer=Manopt.trust_regions, 
                                                    gradient=constrained_optimal_value_Euclidean_gradient_analytic,
                                                    kwargs...)
    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)

    f(M, v) = constrained_optimal_value(target, pert, A, v; regularization)

    function g(M, v)
        gr = gradient(target, pert, A, v; regularization)
        return project(M, v, gr)
    end

    x = optimizer(M, f, g, x0; kwargs...)
end

using Optim

function nearest_unstable_optim(target, pert, A, x0; regularization=0.0,
    gradient=constrained_optimal_value_Euclidean_gradient_analytic,
    kwargs...)

    f(v) = constrained_optimal_value(target, pert, A, v; regularization)
    g(v) = gradient(target, pert, A, v; regularization)
    
    res = optimize(f, g, x0, 
                    inplace=false,
                    Optim.LBFGS(manifold=Optim.Sphere(), m=20), 
                    Optim.Options(iterations=1_000))
    @show res
    return res.minimizer
end



function nearest_unstable_penalty_method!(target, pert, A, x;
                        optimizer=Manopt.quasi_Newton!,
                        gradient=constrained_optimal_value_Euclidean_gradient_analytic,
                        outer_iterations=30, 
                        starting_regularization=1.,
                        regularization_damping = 0.75, kwargs...)

    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)
    regularization = starting_regularization

    E, lambda = constrained_minimizer(target, pert, A, x; regularization)
    df = DataFrame()
    df.outer_iteration_number = [0]
    df.regularization = [regularization]
    df.inner_iterations = [0]
    df.f = [constrained_optimal_value(target, pert, A, x)]
    df.f_reg = [constrained_optimal_value(target, pert, A, x; regularization)]
    df.f_heuristic = [NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]]
    df.constraint_violation = [norm((A+E)*x - x*lambda)]
    
    for k = 1:outer_iterations
        @show k

        E, lambda = constrained_minimizer(target, pert, A, x; regularization)
        # @show original_function_value = constrained_optimal_value(target, A, x)
        # @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(target, A, x)[2]
        # @show constraint_violation = norm((A+E)*x - x*lambda)
        # @show regularization
        # @show k

        f(M, v) = constrained_optimal_value(target, pert, A, v; regularization)

        function g(M, v)
            gr = gradient(target, pert, A, v; regularization)
            return project(M, v, gr)
        end
    
        R = optimizer(M, f, g, x; return_state=true, record=[:Iteration], kwargs...)
        E, lambda = constrained_minimizer(target, pert, A, x; regularization)
        
        # populate results
        push!(df, 
            [k, regularization, length(get_record(R)), 
            constrained_optimal_value(target, pert, A, x),
            constrained_optimal_value(target, pert, A, x; regularization),
            NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2],
            norm((A+E)*x - x*lambda),
            ]
        )

        regularization = regularization * regularization_damping
    end

    return x, df
end


function nearest_unstable_augmented_Lagrangian_method!(target, pert, A, x; optimizer=Manopt.quasi_Newton!,
                                                    gradient=reduced_augmented_Lagrangian_Euclidean_gradient_analytic,
                                                    outer_iterations=60,
                                                    starting_regularization=1., 
                                                    regularization_damping = 0.8,
                                                    kwargs...)
    n = size(A,1)
    M = Manifolds.Sphere(n-1, ℂ)
    y = zero(x)
    regularization = starting_regularization

    E, lambda = reduced_augmented_Lagrangian_minimizer(target, pert, A, x, y; regularization)
    df = DataFrame()
    df.outer_iteration_number = [0]
    df.regularization = [regularization]
    df.inner_iterations = [0]
    df.f = [constrained_optimal_value(target, pert, A, x)]
    df.f_reg = [constrained_optimal_value(target, pert, A, x; regularization)]
    df.f_heuristic = [NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]]
    df.constraint_violation = [norm((A+E)*x - x*lambda)]
    df.normy = [norm(y)]
    df.augmented_Lagrangian = [reduced_augmented_Lagrangian(target, pert, A, x, y; regularization) - regularization*norm(y)^2]
    
    for k = 1:outer_iterations        
        @show k        
        f(M, v) = reduced_augmented_Lagrangian(target, pert, A, v, y; regularization)
        function g_zygote(M, v)
            gr = gradient(target, pert, A, v, y; regularization)
            return project(M, v, gr)
        end

        R = optimizer(M, f, g_zygote, x; return_state=true, record=[:Iteration], kwargs...)
        E, lambda = reduced_augmented_Lagrangian_minimizer(target, pert, A, x, y; regularization)
        push!(df,
            [k, regularization, length(get_record(R)), 
            constrained_optimal_value(target, pert, A, x),
            constrained_optimal_value(target, pert, A, x; regularization),
            NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2],
            norm((A+E)*x - x*lambda),
            norm(y),
            reduced_augmented_Lagrangian(target, pert, A, x, y; regularization) - regularization*norm(y)^2
            ]
        )

        #TODO: compare accuracy of this formula with the vanilla update y .= y + (1/regularization) * ((A+E)*x - x*lambda)
        # y .= inverse_quadratic_form_matvec(pc, A*x + regularization*y - x*lambda)
        y .= y + (1/regularization) * ((A+E)*x - x*lambda)
        regularization = regularization * regularization_damping

    end

    return df
end

function nearest_unstable_augmented_Lagrangian_method_optim(target, pert, A, x0;
        gradient=reduced_augmented_Lagrangian_Euclidean_gradient_analytic,
        outer_iterations=30,
        starting_regularization=1., 
        regularization_damping = 0.75,
        memory_parameter=20,
        verbose=true,
        kwargs...)

    y = zero(x0)  # this will be updated anyway
    regularization = starting_regularization
    x = copy(x0)
    for k = 1:outer_iterations
        if verbose
            @show augmented_Lagrangian = reduced_augmented_Lagrangian(target, pert, A, x, y; regularization) - regularization * norm(y)^2
            @show regularization
            @show k
        end

        # We start with a dual gradient ascent step from x0 to get a plausible y0
        # dual gradient ascent.
        E, lambda = reduced_augmented_Lagrangian_minimizer(target, pert, A, x, y; regularization)
        y .= y + (1/regularization) * ((A+E)*x - x*lambda)
        if verbose
            @show constraint_violation = norm((A+E)*x - x*lambda)
            @show original_function_value = constrained_optimal_value(target, pert, A, x)
            @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]
        end
        f(v) = reduced_augmented_Lagrangian(target, pert, A, v, y; regularization)
        g(v) = gradient(target, pert, A, v, y; regularization)
        
        res = optimize(f, g, x0, 
                        inplace=false,
                        Optim.LBFGS(manifold=Optim.Sphere(), m=memory_parameter), 
                        Optim.Options(; kwargs...))
        if verbose
            @show res
        end

        x .= res.minimizer
        regularization = regularization * regularization_damping
    end
    E, lambda = reduced_augmented_Lagrangian_minimizer(target, pert, A, x, y; regularization)
    if verbose
        @show constraint_violation = norm((A+E)*x - x*lambda)
        @show original_function_value = constrained_optimal_value(target, pert, A, x)
        @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]
    end
    return x
end


"""
    v = fix_unfeasibility!(target, pert, A, v)

Makes sure that the least-squares problem is solvable, i.e., that m2[i] == 0 => v[i] = 0.,
by inserting more zeros in v. This function is only needed if lambda != 0, otherwise the problem
is always solvable.
"""
function fix_unfeasibility!(target, pert, A, v)
    while(true)
        pc = precompute(pert, v, 0.; warn=false)
        m2inv = pc
        to_zero = (m2inv .== 0) .& (v .!= 0)
        if !any(to_zero)
            break
        end
        v[to_zero] .= 0.
    end
end

"""
    v = insert_zero_heuristic!(target, pert, A, v)

    Modifies v to insert zero values instead of certain entries, hoping to reduce the objective function (but not guaranteed).

    By default, when altering a row i to be solvable, this will zero out not only the entries corresponding to P[i,:]
    but also the diagonal entry v[i], which appears in lambda*I.
    
    This is skipped only if `isa(target, Singular)`. Even if one is interested in certain other targets 
    (e.g. NonHurwitz) it might make sense to try the function with target==Singular() to see if the objective value improves.
"""
function insert_zero_heuristic!(target, pert::ComplexSparsePerturbation, A, v)
    Av = ConstantMatrixProduct(A)(v)
    pc = precompute(pert, v, 0.; warn=false)
    lambda = lambda_opt(target, pert, Av, v, pc)

    z = v * lambda - Av
    m2 = ConstantMatrixProduct(pert.P)(abs2.(v))
    lstsq_smallness = m2 + abs2.(z)
    _, i = findmin(x -> x==0 ? Inf : x,  lstsq_smallness)
    v[pert.P[i,:] .!= 0.] .= 0.
    if !isa(target, Singular)
        fix_unfeasibility!(target, pert, A, v)
    end
end
"""
    v, fval = heuristic_zeros(target, pert, A, v_)

    Tries to replace with zeros some entries of v_ (those corresponding to small entries of m2), to get a lower 
    value fval for constrained_optimal_value. Keeps adding zeros iteratively.
"""
function heuristic_zeros(target, pert::ComplexSparsePerturbation, A, v_)
    v = copy(v_)
    bestval = constrained_optimal_value(target, pert, A, v)
    bestvec = copy(v)
    for k = 1:length(v)
        insert_zero_heuristic!(target, pert, A, v)
        curval = constrained_optimal_value(target, pert, A, v)
        if iszero(curval)
            break
        end
        if curval < bestval
            bestval = curval
            bestvec .= v
        end
    end
    return bestvec, bestval
end
heuristic_zeros(target, pert::GeneralPerturbation, A, v_) = (NaN, NaN)  # not implemented yet

end # module
