module NearestUnstableMatrix

using LinearAlgebra
using ForwardDiff
using SparseArrays
using DataFrames

using ChainRulesCore
using ChainRulesCore: ProjectTo, @not_implemented, @thunk
using ManifoldDiff: riemannian_Hessian

import Manifolds: project


export minimizer, optimal_value, minimizer_AplusE, compute_M,
    realgradient, realgradient_zygote,
    InsideDisc, OutsideDisc, NonHurwitz, NonSchur,  RightOf, Singular, PrescribedValue,
    precompute, ComplexSparsePerturbation, GeneralPerturbation, UnstructuredPerturbation, unstructured_perturbation,
    nearest_unstable!, nearest_unstable_penalty_method!,
    heuristic_zeros,
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
function unstructured_perturbation(m, n=m) # TODO: refactor as GeneralPerturbation(pert::UnstructuredPerturbation)
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
[A0' A1' ... Ad'      ]
[   A0' A1' ... Ad'   ]
[      ....  ...      ] * z
[      A0' A1' ... Ad']
```
"""
adjoint_product(A, v) = A' * v
function adjoint_product(A::MatrixPolynomial, z)
    n = size(A, 2)
    d = size(A, 3) - 1
    k = Int(length(z) / n)
    # reshape(reduce(vcat, sum(A[:,:,i+1]' * z[(h+i)*n+1:(h+i+1)*n] for i=0:d) for h=0:k-d-1), (n,1,k-d))
    reduce(vcat, sum(A[:,:,i+1]' * z[(h+i)*n+1:(h+i+1)*n] for i=0:d) for h=0:k-d-1)
end

"""
    compute_M(pert::GeneralPerturbation, v)

Compute the rhs of the constraint M*delta = r.
"""
function compute_M(pert::GeneralPerturbation, v)
    return reduce(hcat, product(E,v) for E in pert.EE)
end
function compute_M(pert::ComplexSparsePerturbation, v)
    II, JJ, _ = findnz(sparse(pert.P))
    return sparse(II, 1:length(II), v[JJ])
end

m_row_transpose(i,d,v) = reduce(vcat, i-j in 1:size(v,3) ? v[:,:,i-j] : zero(v[:,:,1]) for j=0:d)
m_transpose(d,v) = reduce(hcat, m_row_transpose(i,d,v) for i=1:size(v,3)+d)
compute_M(pert::UnstructuredPerturbation, v::MatrixPolynomial) = transpose(m_transpose(pert.d, v))
compute_M(pert::UnstructuredPerturbation, v::AbstractVector) = compute_M(pert, reshape(v, (pert.n, 1, :)))

inftozero(x) = ifelse(isinf(x), zero(x), x)
"""
    U, D, VS = precompute(pert, v; regularization)

Compute an SVD of M, and returns U, D = (S^2+regularization*I)⁻¹, V*S.
These are quantities that can be quickly computed also for a ComplexSparsePerturbation.
"""
function precompute(pert, v; regularization=0.0, M = compute_M(pert, v))
    pc = svd(M)
    return pc.U, inv(Diagonal(pc.S)^2 + regularization*I), pc.V* Diagonal(pc.S)
end
function precompute(pert::ComplexSparsePerturbation, v; regularization=0.0, M = compute_M(pert, v))
    D = Diagonal(1 ./ (ConstantMatrixProduct(pert.P)(abs2.(v)) .+ regularization))
    VS = compute_M(pert, v)'
    return I, D, VS
end

"""
    svdmin(pert, v; M = compute_M(pert, v))

Computes min(svd(M))
"""
function svdmin(pert, v; M = compute_M(pert, v))
    U, D, VS = precompute(pert, v; regularization=0.0, M)
    return sqrt(minimum(diag(D)))
end

# TODO: we need to rethink the compute_r interface. This method needs too many parameters
# and is too tightly coupled with the rest of the computation.
"""
    compute_r(target, pert, Av, v, pc; regularization=0.0)

Compute the rhs of the constraint M*delta = r, and returns the pair (r, lambda)
"""
function compute_r(target::Region, pert, Av, v, pc; regularization=0.0)
    U, D, VS = pc
    Uv = U'*v
    UAv = U'*Av
    nv0 = D * Uv
    denom = nv0' * Uv
    numer = nv0' * UAv
    lambda = project(target, numer / denom)
    return v*lambda - Av, lambda
    # TODO: we can actually now compute U'*r, which is used in some cases. Can we refactor to return it?
end
compute_r(target::PrescribedValue{lambda}, pert, Av, v, pc; regularization=0.0) where lambda = (v*lambda - Av, convert(eltype(v), lambda))
compute_r(target::Singular, pert, Av, v, pc; regularization=0.0) = (-Av, zero(eltype(v)))
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
    return permutedims(reshape(delta, (pert.n, pert.d+1, pert.m)), (3,1,2))
end

function compute_E(pert::ComplexSparsePerturbation, delta)
    II, JJ, _ = findnz(sparse(pert.P))
    return sparse(II, JJ, delta)
end

"""
    E, lambda = minimizer(target, pert A, v, y; regularization=0.0)

Computes the argmin corresponding to `optimal_value`
"""
function minimizer(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v; regularization)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    U, D, VS = pc
    t = U' * r
    # TODO: tweak compute_r to return U'*r directly
    delta = VS * (D * t)
    E = compute_E(pert, delta)
    return E, lambda
end

"""
    optval = optimal_value(target, pert, A, v, y=nothing; regularization=0.0, lambda=nothing)

Computes optval = min_{E ∈ pert} ||E||^2 s.t. (A+E)v = v*λ

λ is chosen (inside the target region or on its border) to minimize `optimal_value(A, v, λ)`

If y is given, computes the reduced augmented Lagrangian (augmented Lagrangian) - ε ||y||².
"""
function optimal_value(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    pc = precompute(pert, v; regularization)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    U, D, VS = pc
    t = U' * r
    return sum(D * abs2.(t))
end

"""
    Simpler, more unstable implementation but easier to autodiff since it doesn't use the SVD.
"""
function optimal_value_naif(target::Singular, pert::UnstructuredPerturbation, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A,v)
    else
        Av = product(A,v) + regularization * y
    end
    M = compute_M(pert, v)
    MM = kron(M, Matrix(I, pert.m, pert.m))
    r = -Av
    return real((r' * ((MM*MM' + regularization*I) \ r))[1,1])
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
    pc = precompute(pert, v; regularization)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    U, D, VS = pc
    t = U' * r
    return hcat(D, D * abs2.(t))
end

function Euclidean_gradient_zygote(target, pert, A, v, y=nothing; regularization=0.0)
    return first(realgradient_zygote(x -> optimal_value(target, pert, A, x, y; regularization), v))
end

"""
    M, z, AplusE, pc, lambda = gradient_helper(target, pert, A, v, y=nothing; regularization=0.0)

Return various intermediate results for gradients and Hessians.
"""
function gradient_helper(target, pert, A, v, y=nothing; regularization=0.0)
    if y===nothing
        Av = product(A, v)
    else
        Av = product(A, v) + regularization * y
    end
    M = compute_M(pert, v)
    pc = precompute(pert, v; regularization, M)
    r, lambda = compute_r(target, pert, Av, v, pc; regularization)
    U, D, VS = pc
    Dt = D * (U' * r)
    delta = VS * Dt
    z = U * Dt
    
    AplusE = A + compute_E(pert, delta)
    return M, z, AplusE, pc, lambda
end

"""
    Euclidean_gradient_analytic(target, pert, A, v, y=nothing; regularization=0.0)

Gradient of the objective function
"""
Euclidean_gradient_analytic(target, pert, A, v, y=nothing; regularization=0.0) = 
        Euclidean_gradient_analytic(gradient_helper(target, pert, A, v, y; regularization))
function Euclidean_gradient_analytic(gradient_helper)
    M, z, AplusE, pc, lambda = gradient_helper
    vecz = transpose(z)[:]  # undoes matrix structure for the UnstructuredPerturbation, a no-op otherwise
    if iszero(lambda)
        return -2 * adjoint_product(AplusE, vecz)
    else
        return 2(z*lambda' - adjoint_product(AplusE, vecz))
    end
end

"""
    Euclidean_Hessian_product_analytic(w, target, pert, A, v, y=nothing; regularization = 0.0)

Compute the product H*w, where w is the Euclidean Hessian
"""
Euclidean_Hessian_product_analytic(w, target, pert, A, v, y=nothing; regularization = 0.0) = 
        Euclidean_Hessian_product_analytic(w, pert, gradient_helper(target, pert, A, v, y; regularization); regularization)
function Euclidean_Hessian_product_analytic(w, pert, gradient_helper; regularization=0.0)
    M, z, AplusE, pc, lambda = gradient_helper
    @assert lambda == 0 # for now
    dM = compute_M(pert, w)
    rightpart = transpose(reshape(-product(AplusE, w), (:, size(M, 1)))) - M * (dM'*z)
    U, D, VS = pc
    dz = U * (D * (U' * rightpart))
    dAplusE = compute_E(pert, (M'*dz + dM'*z)[:])
    vecz = transpose(z)[:]
    vecdz = transpose(dz)[:]
    return 2(-adjoint_product(dAplusE, vecz) - adjoint_product(AplusE, vecdz))
end

Euclidean_Hessian_product_and_gradient_analytic(w, target, pert, A, v, y=nothing; regularization = 0.0) =
    Euclidean_Hessian_product_and_gradient_analytic(w, pert, gradient_helper(target, pert, A, v, y; regularization); regularization)
function Euclidean_Hessian_product_and_gradient_analytic(w, pert, gradient_helper; regularization=0.0)
    return Euclidean_Hessian_product_analytic(w, pert, gradient_helper; regularization), Euclidean_gradient_analytic(gradient_helper)
end

"""
    g = realgradient(f, cv)

Computes the Euclidean gradient of a function f: C^n -> C^n (seen as C^n ≡ R^{2n}), using forward-mode AD
"""
function realgradient(f, cv::AbstractVector)
   n = length(cv)
   gr = ForwardDiff.gradient(x -> f(x[1:n] + 1im * x[n+1:end]), [real(cv); imag(cv)]) 
   return gr[1:n] + 1im * gr[n+1:end]
end
function realhessian(f, cv::AbstractVector)
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
    nearest_unstable!(target, pert, A, x0; regularization=0.0, 
        optimizer=Manopt.trust_regions!), gradient=Euclidean_gradient_analytic, kwargs...

Compute v such that (A+E)v = v*lambda, lambda is inside the target region, and ||E||_F is a (local) minimum.
Additional keyword arguments are passed to the optimizer (for `debug`, `stopping_criterion`, etc.).

x0 is overwritten with the result v. Return `return_state` from Manopt, typically with just the number of iterations.
"""
function nearest_unstable!(target, pert, A, x, y=nothing; regularization=0.0,
                                                    optimizer=Manopt.quasi_Newton!,                                                     
                                                    gradient=Euclidean_gradient_analytic,
                                                    use_Hessian=false,
                                                    kwargs...)
    n = length(x)
    M = Manifolds.Sphere(n-1, eltype(x)<:Complex ? ℂ : ℝ)

    f(M, v) = optimal_value(target, pert, A, v, y; regularization)

    function grad(M, v)
        gr = gradient(target, pert, A, v, y; regularization)
        return project(M, v, gr)
    end
    function hess(M, v, w)
        eh, eg = Euclidean_Hessian_product_and_gradient_analytic(w, target, pert, A, v, y; regularization)
        return riemannian_Hessian(M, v, eg, eh, w)
    end

    if use_Hessian
        R = optimizer(M, f, grad, hess, x; return_state=true, record=[:Iteration], kwargs...)
    else
        R = optimizer(M, f, grad, x; return_state=true, record=[:Iteration], kwargs...)
    end
    return R
end


"""
    nearest_unstable_penalty_method!(target, pert, A, x, y; kwargs...)

Solve the nearest unstable problem with a penalty method, optionally with 
an augented Lagrangian term (if `y` is given; `y=zero(A*x)` is a reasonable starting value).

`x` and `y` contain initial values, and are overwritten in-place.
"""
function nearest_unstable_penalty_method!(target, pert, A, x, y=nothing; optimizer=Manopt.quasi_Newton!,
                                                    gradient=Euclidean_gradient_analytic,
                                                    use_Hessian=false,
                                                    outer_iterations=60,
                                                    starting_regularization=1., 
                                                    regularization_damping = 0.8,
                                                    adjust_speed=false,                                                    
                                                    verbose=true,
                                                    kwargs...)

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
    df.normy = [y===nothing ? nothing : norm(y)]
    df.augmented_Lagrangian = [y===nothing ? nothing : optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2]
    df.minsvd = [svdmin(pert, x)]

    for k = 1:outer_iterations
        if any(isnan.(x))
            break
        end
        if verbose
            @show k
        end

        inner_its = 0 # overwritten after each iteration
        try
            R = nearest_unstable!(target, pert, A, x, y; regularization, optimizer, gradient, use_Hessian, kwargs...)
            inner_its = length(get_record(R))
        catch e
            if isa(e, InterruptException)
                @info "Got interrupt, exiting"
                break
            else 
                rethrow()
            end
        end

        E, lambda = minimizer(target, pert, A, x, y; regularization)
        push!(df,
            [k, regularization, inner_its, 
            optimal_value(target, pert, A, x),
            optimal_value(target, pert, A, x; regularization),
            NearestUnstableMatrix.heuristic_zeros(target, pert, A, x)[2],
            norm(constraint(target, pert, A, E, x, lambda)),
            y === nothing ? nothing : norm(y),
            y === nothing ? nothing : optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2,
            svdmin(pert, x)
            ]
        )
            
        #TODO: compare accuracy of this formula with the vanilla update y .= y + (1/regularization) * (constraint(target, pert, A, E, x, lambda))
        # y .= pc .* (constraint(target, pert, A, E, x, lambda) + regularization*y)
        if verbose
            @show constraint_violation = norm(constraint(target, pert, A, E, x, lambda))
        end
        if !(y===nothing) 
            y .= y + (1/regularization) * (constraint(target, pert, A, E, x, lambda))
            if verbose
                @show norm(y)
                @show lagrangian = optimal_value(target, pert, A, x, y; regularization) - regularization*norm(y)^2
            end
        end

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
        if verbose
            @show regularization
        end
    end    
    return df
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
        pc = precompute(pert, v; regularization=0.0)
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
    pc = precompute(pert, v; regularization=0.0)
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
