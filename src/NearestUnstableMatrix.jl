module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff
using SparseArrays
using DataFrames

using ChainRulesCore
using ChainRulesCore: ProjectTo, @not_implemented, @thunk
import Manifolds: project

export minimizer, optimal_value, minimizer_AplusE, compute_M,
    realgradient, realgradient_zygote,
    InsideDisc, OutsideDisc, NonHurwitz, NonSchur,  RightOf, Singular, PrescribedValue,
    precompute, ComplexSparsePerturbation, GeneralPerturbation, UnstructuredPerturbation, unstructured_perturbation,
    nearest_unstable, heuristic_zeros,
    ConstantMatrixProduct,
    project, toeplitz_perturbation, grcar,
    MatrixPolynomial
    
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
struct RightOf{r} <: Region
end
RightOf(r) = RightOf{r}()
const NonHurwitz = RightOf{0.0}

struct PrescribedValue{v} <: Region
end
PrescribedValue(v) = PrescribedValue{v}()
"""
For a matrix polynomial, we take "singular" to mean "singular polynomial", not "an eigenvalue equal to 0",
since the latter problem doesn't make much sense to consider: to solve it, it is enough to compute 
the distance of A₀ from singular matrices, ignoring the other coefficients.
"""
const Singular = PrescribedValue{0.0}

function project(::OutsideDisc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a < r, lambda/a*r, lambda)
end
function project(::InsideDisc{r}, lambda) where r
    a = abs(lambda)
    return ifelse(a > r, lambda/a*r, lambda)
end
function project(::RightOf{r}, lambda) where r
    return ifelse(real(lambda)<r, lambda-real(lambda)+r, lambda)
end
function project(::PrescribedValue{v}, lambda) where v
    return v
end

const MatrixPolynomial{T} = Array{T, 3} where T



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
    return GeneralPerturbation(EE)
end

"""
    struct UnstructuredPerturbation <: PerturbationStructure end

Type to represent an unstructured perturbation (all entries of Δ can vary independently).

In this case, we take advantage of the structure of the matrix M and formulate the problem with a multi-column δ.
"""
struct UnstructuredPerturbation <: PerturbationStructure
    m::Int64
    n::Int64
    d::Int64
end
UnstructuredPerturbation(A::MatrixPolynomial) = UnstructuredPerturbation(size(A,1), size(A,2), size(A,3)-1)

"""
    Returns a GeneralPerturbation corresponding to perturbing entries of the matrix polynomial v to make V exactly singular.
"""
function M_perturbation(pert, M::AbstractMatrix)
    vlength = pert.n * (size(M,1) - pert.d)
    id = Matrix(I, vlength, vlength)
    EE = [transpose(compute_M(pert, Float64.(y))) for y in eachcol(id)]
    return GeneralPerturbation(EE)
end

"""
    unstructured_perturbation(m, n=m)
    unstructured_perturbation(m, n=m; degree)

Return a GeneralPerturbation corresponding to an unstructured perturbation (i.e., each entry can be perturbed independently)
of an mxn matrix or a mxn matrix polynomial of given `degree`.
"""
function unstructured_perturbation(m, n=m)
    EE = collect(sparse([i],[j], [1.], m, n) for j=1:n for i=1:m)
    return GeneralPerturbation(EE)
end
function unstructured_perturbation(m, n, degree)
    EE = MatrixPolynomial{Float64}[]
    for d = 0:degree
        for j = 1:n
            for i = 1:m
                E = zeros(m, n, degree+1)
                E[i, j, d+1] = 1.
                push!(EE, E)
            end
        end
    end
    return GeneralPerturbation(EE)
end
unstructured_perturbation(A::AbstractMatrix) = unstructured_perturbation(size(A,1), size(A,2))
unstructured_perturbation(A::MatrixPolynomial) = unstructured_perturbation(size(A,1), size(A,2), size(A,3)-1)

"""
    toeplitz_perturbation(n)
    toeplitz_perturbation(A)

Return a GeneralPerturbation that allows (complex) Toeplitz perturbations of a n×n matrix.
"""
function toeplitz_perturbation(n::Number, indices=-(n-1):(n-1))
    EE = collect(1/sqrt(n-abs(k)) * diagm(k => ones(n-abs(k))) for k in indices)
    return GeneralPerturbation(EE)
end
toeplitz_perturbation(A::AbstractMatrix) = toeplitz_perturbation(size(A,1))
toeplitz_perturbation(A::AbstractMatrix, indices) = toeplitz_perturbation(size(A,1), indices)

"""
    grcar(n)

Compute the nxn Grcar matrix
"""
function grcar(n)
    return diagm(-1=>-ones(n-1), 0=>ones(n), 1=>ones(n-1), 2=>ones(n-2), 3=>ones(n-3))
end


"""
    sum_pairs(m, n, p)

Return pairs from 1:m 1:n that sum to p. Used for convolution / polynomial products.
"""
sum_pairs(m, n, p) = ((i, p-i) for i in max(1, p-n):min(m, p-1))
"""
    product_coefficient(A, B, d)

Return the degree-d coefficient of the product A(x)*B(x), given coefficients
of matrix polynomials A, B in a 3-dimensional array: A[:,:,1] is the constant term.
"""
product_coefficient(A, B, d) = sum(A[:,:,i]*B[:,:,j] for (i,j)=sum_pairs(size(A,3), size(B,3), d+2))

"""
    product(A, v)

Returns the standard matrix product A*v typically, but form matrix polynomials it returns
    the coefficients of A*v stacked.
"""
product(A, v) = ConstantMatrixProduct(A)(v)
function product(A::MatrixPolynomial, v::MatrixPolynomial)
    reduce(vcat, product_coefficient(A, v, d) for d=0:size(A,3)+size(v,3)-2)
end
product(A::MatrixPolynomial, v::AbstractVector) = product(A, reshape(v, (size(A,2), 1, :)))

"""
    adjoint_product(A, z)

Returns A'*z typically, but for matrix polynomials it returns the stacked product by shifted versions of [A0 A1 ... Ad]:
```
[A0 A1 ... Ad      ]
[   A0 A1 ... Ad   ]
[      ....        ] * z
[      A0 A1 ... Ad]
```
"""
adjoint_product(A, v) = A' * v
function adjoint_product(A::MatrixPolynomial, z)
    n = size(A, 2)
    d = size(A, 3) - 1
    k = Int(length(z) / n)
    reshape(reduce(vcat, sum(A[:,:,i+1]' * z[(h+i)*n+1:(h+i+1)*n] for i=0:d) for h=0:k-d-1), (n,1,k-d))
end

"""
    compute_M(pert::GeneralPerturbation, v)

Compute the rhs of the constraint M*delta = r.
"""
function compute_M(pert::GeneralPerturbation, v)
    return reduce(hcat, product(E,v) for E in pert.EE)
end
# the following is not the most efficient implementation, but we don't plan to use it in tight loops
compute_M(pert::ComplexSparsePerturbation, v) = compute_M(GeneralPerturbation(pert), v)

m_row_transpose(i,d,v) = reduce(vcat, i-j in 1:size(v,3) ? v[:,:,i-j] : zero(v[:,:,1]) for j=0:d)
m_transpose(d,v) = reduce(hcat, m_row_transpose(i,d,v) for i=1:size(v,3)+d)
compute_M(pert::UnstructuredPerturbation, v::MatrixPolynomial) = transpose(m_transpose(pert.d, v))
compute_M(pert::UnstructuredPerturbation, v::AbstractVector) = compute_M(pert, reshape(v, (pert.n, 1, :)))

inftozero(x) = ifelse(isinf(x), zero(x), x)
"""
    pc = precompute(pert, v, regularization)

Computes "something related" to the inverse of the weighting matrix M.

* For ComplexSparsePerturbation, it is the inverse of the weight vector, 
m2inv[i] = 1 / (||d_i||^2 + epsilon), where d_i is the vector with d_i=1 iff A[i,j]≠0 and 0 otherwise.
Moreover, the method adjusts the vector such that 1/0=Inf is replaced by 0, to avoid NaNs in further computations.

* For GeneralPerturbation, it is SVD(M_v)
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


# TODO: we need to rethink the compute_r interface. This method needs too many parameters
# and is too tightly coupled with the rest of the computation.
"""
    compute_r(target, pert, Av, v, pc; regularization=0.0)

Compute the rhs of the constraint M*delta = r, and returns the pair (r, lambda)
"""
function compute_r(target::Region, pert::ComplexSparsePerturbation, Av, v, pc; regularization=0.0)
    nv = v .* pc
    denom = nv' * v
    numer = nv' * Av
    lambda = project(target, numer / denom)
    return v*lambda - Av, lambda
end

function compute_r(target::Region, pert::GeneralPerturbation, Av, v, pc; regularization=0.0)
    Uv = pc.U'*v
    UAv = pc.U'*Av
    nv0 = Diagonal(pc.S.^2 .+ regularization) \ Uv
    denom = nv0' * Uv
    numer = nv0' * UAv
    lambda = project(target, numer / denom)
    return v*lambda - Av, lambda
    # TODO: we can actually now compute U'*r, which is used in some cases. Can we refactor to return it?
end
compute_r(target::PrescribedValue{lambda}, pert::ComplexSparsePerturbation, Av, v, pc; regularization=0.0) where lambda = (v*lambda - Av, convert(eltype(v), lambda))
compute_r(target::PrescribedValue{lambda}, pert::GeneralPerturbation, Av, v, pc; regularization=0.0) where lambda = (v*lambda - Av, convert(eltype(v), lambda))
compute_r(target::PrescribedValue{lambda}, pert::UnstructuredPerturbation, Av, v, pc; regularization=0.0) where lambda = (v*lambda - Av, convert(eltype(v), lambda))
compute_r(target::Singular, pert::ComplexSparsePerturbation, Av, v, pc; regularization=0.0) = (-Av, zero(eltype(v)))
compute_r(target::Singular, pert::GeneralPerturbation, Av, v, pc; regularization=0.0) = (-Av, zero(eltype(v)))
function compute_r(target::Singular, pert::UnstructuredPerturbation, Av, v, pc; regularization=0.0)
    return (transpose(-reshape(Av, (pert.m,:))), convert(eltype(v), 0.))
end

"""
    compute_E(pert, delta)

Compute the matrix / matrix polynomial E from its basis representation
"""
function compute_E(pert::GeneralPerturbation, delta)
    return reduce(+, E*m for (E,m) in zip(pert.EE, delta))
end
function compute_E(pert::UnstructuredPerturbation, delta)
    # delta = [transpose(E0); transpose(E1); ... ; transpose(Ed)]
    return permutedims(reshape(delta, (pert.m, pert.n, pert.d+1)), (3,1,2))
end

"""
    E, lambda = minimizer(target, pert A, v, y; regularization=0.0)

Computes the argmin corresponding to `optimal_value`
"""
function minimizer(target, pert::ComplexSparsePerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    t = r .* pc
    E = t .* (v' .* pert.P)  # the middle .* broadcasts column * row
    return E, lambda
end
function minimizer(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    M = compute_M(pert, v)
    pc = svd(M)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    t = pc.U' * r
    # TODO: tweak compute_r to return U'*r directly
    delta = pc.V * (Diagonal(pc.S ./ (pc.S.^2 .+ regularization)) * t)
    E = compute_E(pert, delta)
    return E, lambda
end

"""
    `AplusE, lambda = minimizer_AplusE(target, pert, alpha, v; regularization=0.0)`

Computes the value of A+E
"""
function minimizer_AplusE(target, pert::ComplexSparsePerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    nv = pc .* r
    t1 = (v*lambda) .* pc
    vP = v' .* pert.P  # matrix with the nonzero structure of A but elements conj(v_j)
    E1 = t1 .* vP # broadcasting
    s2 = pert.P * abs2.(v)
    nonpert = Av ./ s2
    projA = A - nonpert .* vP # this projects A on Im(pc.V)^⟂
    # we subtract this projection before the next addition in the hope of getting
    # exact zeros where needed, and avoid losing accuracy.
    minimizer_AplusE = projA + E1 + (Diagonal(regularization ./ (s2 .+ regularization) ./ s2) *Av) .* vP
    return minimizer_AplusE, lambda, nv
end

"""
Alternate formula for the gradient, in the hope of getting better accuracy (but it didn't work in the end)
"""
function gradient_alternative(target, pert, A, v, y=nothing; regularization=0.0)
    AplusE, lambda, nv = minimizer_AplusE(target, pert, A, v, y; regularization)
    return 2(nv*lambda' - AplusE'*nv)
end

"""
    optval = optimal_value(target, pert, A, v, y=nothing; regularization=0.0, lambda=nothing)

Computes optval = min_{E ∈ pert} ||E||^2 s.t. (A+E)v = v*λ

λ is chosen (inside the target region or on its border) to minimize `optimal_value(A, v, λ)`

If y is given, computes the reduced augmented Lagrangian (augmented Lagrangian) - ε ||y||².
"""
function optimal_value(target, pert::ComplexSparsePerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    optval = sum(abs2.(r) .* pc)
    return optval
end

function optimal_value(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    M = compute_M(pert, v)
    pc = svd(M)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    t = pc.U' * r
    return sum((Diagonal(pc.S)^2 + regularization*I)  \ abs2.(t))
end

"""
Returns a nx2 matrix with the singular values of M in the first column, and the summands that compose optimal_value "along" each singular value
"""
function optimal_vector_entries(target, pert, A, v, y=nothing; regularization=0.)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    M = compute_M(pert, v)
    pc = svd(M)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    t = pc.U' * r
    return hcat(pc.S, Diagonal(pc.S)^2 \ abs2.(t))
end

function Euclidean_gradient_zygote(target, pert, A, v, y=nothing; regularization=0.0)
    return first(realgradient_zygote(x -> optimal_value(target, pert, A, x, y; regularization), v))
end

function Euclidean_gradient_analytic(target, pert::ComplexSparsePerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v, regularization; warn=!isa(target, Singular))
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    nv = -r .* pc
    grad = 2(A'*nv - nv*lambda' - (pert.P' * abs2.(nv)) .* v)
    return grad
end

function Euclidean_gradient_analytic(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    M = compute_M(pert, v)
    pc = svd(M)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    t = pc.U' * r
    delta = pc.V * (Diagonal(pc.S ./ (pc.S.^2 .+ regularization)) * t)
    nv = pc.U * (Diagonal(1 ./ (pc.S.^2 .+ regularization)) * t)
    vecnv = transpose(nv)[:]  # undoes matrix structure for the UnstructuredPerturbation, a no-op otherwise
    AplusE = A + compute_E(pert, delta)
    if iszero(lambda)
        return -2 * adjoint_product(AplusE, vecnv)
    else
        return 2(nv*lambda' - adjoint_product(AplusE, vecnv))
    end
end

"""
    g = realgradient(f, cv)

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
    g = realgradient_zygote(f, cv)

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

function constraint(target::Singular, pert, A, E, x, lambda)
    return product(A+E, x)
end
function constraint(target, pert, A, E, x, lambda)
    return product(A+E, x) - x*lambda
end



using Manifolds, Manopt
"""
    nearest_unstable(target, pert, A, args...; optimizer=Manopt.trust_regions)

Computes v such that (A+E)v = v*lambda, lambda is inside the target region, and ||E||_F is minimal
Additional keyword arguments are passed to the optimizer (for `debug`, `stopping_criterion`, etc.).
"""
function nearest_unstable(target, pert, A, x0; regularization=0.0, 
                                                    optimizer=Manopt.trust_regions, 
                                                    gradient=Euclidean_gradient_analytic,
                                                    kwargs...)
    n = length(x0)
    M = Manifolds.Sphere(n-1, eltype(x0)<:Complex ? ℂ : ℝ)

    f(M, v) = optimal_value(target, pert, A, v; regularization)

    function g(M, v)
        gr = gradient(target, pert, A, v; regularization)
        return project(M, v, gr)
    end

    x = optimizer(M, f, g, x0; kwargs...)
end

using Optim

function nearest_unstable_optim(target, pert, A, x0; regularization=0.0,
    gradient=Euclidean_gradient_analytic,
    kwargs...)

    f(v) = optimal_value(target, pert, A, v; regularization)
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
                        gradient=Euclidean_gradient_analytic,
                        outer_iterations=30, 
                        starting_regularization=1.,
                        regularization_damping = 0.75, kwargs...)

    n = length(x)
    M = Manifolds.Sphere(n-1, eltype(x)<:Complex ? ℂ : ℝ)
    regularization = starting_regularization

    E, lambda = minimizer(target, pert, A, x; regularization)
    df = DataFrame()
    df.outer_iteration_number = [0]
    df.regularization = [regularization]
    df.inner_iterations = [0]
    df.f = [optimal_value(target, pert, A, x)]
    df.f_reg = [optimal_value(target, pert, A, x; regularization)]
    df.f_heuristic = [NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]]
    df.constraint_violation = [norm(constraint(target, pert, A, E, x, lambda))]
    
    for k = 1:outer_iterations
        @show k
        if any(isnan.(x))
            break
        end

        E, lambda = minimizer(target, pert, A, x; regularization)
        # @show original_function_value = optimal_value(target, A, x)
        # @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(target, A, x)[2]
        # @show constraint_violation = norm((A+E)*x - x*lambda)
        # @show regularization
        # @show k

        f(M, v) = optimal_value(target, pert, A, v; regularization)

        function g(M, v)
            gr = gradient(target, pert, A, v; regularization)
            return project(M, v, gr)
        end
    
        R = optimizer(M, f, g, x; return_state=true, record=[:Iteration], kwargs...)
        E, lambda = minimizer(target, pert, A, x; regularization)
        
        # populate results
        push!(df, 
            [k, regularization, length(get_record(R)), 
            optimal_value(target, pert, A, x),
            optimal_value(target, pert, A, x; regularization),
            NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2],
            norm(constraint(target, pert, A, E, x, lambda)),
            ]
        )

        regularization = regularization * regularization_damping
    end

    return df
end


function nearest_unstable_augmented_Lagrangian_method!(target, pert, A, x; optimizer=Manopt.quasi_Newton!,
                                                    gradient=Euclidean_gradient_analytic,
                                                    outer_iterations=60,
                                                    starting_regularization=1., 
                                                    regularization_damping = 0.8,
                                                    adjust_speed=false,
                                                    target_iterations = 800,
                                                    kwargs...)
    n = length(x)
    M = Manifolds.Sphere(n-1, eltype(x)<:Complex ? ℂ : ℝ)
    y = zero(product(A,x))
    regularization = starting_regularization

    E, lambda = minimizer(target, pert, A, x, y; regularization)
    df = DataFrame()
    df.outer_iteration_number = [0]
    df.regularization = [regularization]
    df.inner_iterations = [0]
    df.f = [optimal_value(target, pert, A, x)]
    df.f_reg = [optimal_value(target, pert, A, x; regularization)]
    df.f_heuristic = [NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]]
    df.constraint_violation = [norm(constraint(target, pert, A, E, x, lambda))]
    df.normy = [norm(y)]
    df.augmented_Lagrangian = [optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2]
    
    for k = 1:outer_iterations        
        if any(isnan.(x))
            break
        end
        @show k        
        f(M, v) = optimal_value(target, pert, A, v, y; regularization)
        function g_zygote(M, v)
            gr = gradient(target, pert, A, v, y; regularization)
            return project(M, v, gr)
        end

        R = optimizer(M, f, g_zygote, x; return_state=true, record=[:Iteration], kwargs...)
        E, lambda = minimizer(target, pert, A, x, y; regularization)
        push!(df,
            [k, regularization, length(get_record(R)), 
            optimal_value(target, pert, A, x),
            optimal_value(target, pert, A, x; regularization),
            NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2],
            norm(constraint(target, pert, A, E, x, lambda)),
            norm(y),
            optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2
            ]
        )

        #TODO: compare accuracy of this formula with the vanilla update y .= y + (1/regularization) * (constraint(target, pert, A, E, x, lambda))
        # y .= pc .* (constraint(target, pert, A, E, x, lambda) + regularization*y)
        y .= y + (1/regularization) * (constraint(target, pert, A, E, x, lambda))
        @show norm(y)
        @show constraint_violation = norm(constraint(target, pert, A, E, x, lambda))
        @show lagrangian = optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2

        if adjust_speed
            inner_its = length(get_record(R))
            if inner_its < 300
                regularization = regularization * regularization_damping^2
            elseif inner_its < 1000
                regularization = regularization * regularization_damping
            else
                regularization = regularization * sqrt(regularization_damping)
            end
        else
            regularization = regularization * regularization_damping
        end
        @show regularization
    end

    return df
end

function nearest_unstable_augmented_Lagrangian_method_optim(target, pert, A, x0;
        gradient=Euclidean_gradient_analytic,
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
            @show augmented_Lagrangian = optimal_value(target, pert, A, x, y; regularization) - regularization * norm(y)^2
            @show regularization
            @show k
        end
        if any(isnan.(x))
            break
        end


        # We start with a dual gradient ascent step from x0 to get a plausible y0
        # dual gradient ascent.
        E, lambda = minimizer(target, pert, A, x, y; regularization)
        y .= y + (1/regularization) * ((A+E)*x - x*lambda)
        if verbose
            @show constraint_violation = norm((A+E)*x - x*lambda)
            @show original_function_value = optimal_value(target, pert, A, x)
            @show heuristic_value = NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2]
        end
        f(v) = optimal_value(target, pert, A, v, y; regularization)
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
    E, lambda = minimizer(target, pert, A, x, y; regularization)
    if verbose
        @show constraint_violation = norm((A+E)*x - x*lambda)
        @show original_function_value = optimal_value(target, pert, A, x)
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

Modify `v` to insert zero values instead of certain entries, hoping to reduce the objective function (but not guaranteed).

By default, when altering a row i to be solvable, this will zero out not only the entries corresponding to P[i,:]
but also the diagonal entry v[i], which appears in lambda*I.

This is skipped only if `isa(target, Singular)`. Even if one is interested in certain other targets 
(e.g. NonHurwitz) it might make sense to try the function with target==Singular() to see if the objective value improves.
"""
function insert_zero_heuristic!(target, pert::ComplexSparsePerturbation, A, v)
    Av = ConstantMatrixProduct(A)(v)
    pc = precompute(pert, v, 0.; warn=false)
    r, lambda = compute_r(target, pert, Av, v, pc)
    m2 = ConstantMatrixProduct(pert.P)(abs2.(v))
    lstsq_smallness = m2 + abs2.(r)
    _, i = findmin(x -> x==0 ? Inf : x,  lstsq_smallness)
    v[pert.P[i,:] .!= 0.] .= 0.
    if !isa(target, Singular)
        fix_unfeasibility!(target, pert, A, v)
    end
end
"""
    v, fval = heuristic_zeros(target, pert, A, v_)

Try to replace with zeros some entries of v_ (those corresponding to small entries of m2), to get a lower 
value fval for `optimal_value`. Keeps adding zeros iteratively.
"""
function heuristic_zeros(target, pert::ComplexSparsePerturbation, A, v_)
    v = copy(v_)
    bestval = optimal_value(target, pert, A, v)
    bestvec = copy(v)
    for k = 1:length(v)
        insert_zero_heuristic!(target, pert, A, v)
        curval = optimal_value(target, pert, A, v)
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
heuristic_zeros(target, pert, A, v_) = (NaN, NaN)  # not implemented yet

end # module
