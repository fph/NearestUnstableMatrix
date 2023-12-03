using NearestUnstableMatrix
using Test
using Random
using LinearAlgebra
using SparseArrays
using Manifolds, Manopt

using NearestUnstableMatrix: NonSchur, project, precompute

@testset "project" begin
    @test project(OutsideDisc(2.3), 1) == 2.3
    @test project(OutsideDisc(2.3), 5) == 5
    @test project(OutsideDisc(2.3), 5im) == 5im
    @test project(RightOf(2.3), 5im) == 2.3 + 5im
    @test project(RightOf(2.3), 5+5im) == 5 + 5im
end

@testset "precompute" begin
    P = ComplexSparsePerturbation([0 1; 1 0])
    v = [2; 0]
    @test isequal(precompute(P, v, 1.0), [1.0; 0.2])
    @test isequal(precompute(P, v, 0.0), [0.0; 0.25])
    
end

@testset "Function values and derivatives" begin

    # Simple Singular() case
    A = [1. 2; 3 4]
    n = size(A, 1)
    v = ComplexF64.([3/5;4/5])
    target = Singular()
    P = ComplexSparsePerturbation(A.!=0)
    E, lambda = constrained_minimizer(target, P, A, v)
    @test E ≈ [-1.32 -1.76; -3 -4]
    @test lambda == 0
    @test constrained_optimal_value(target, P, A, v) ≈ norm(E)^2
    AplusE, lambda2 = constrained_AplusE(target, P, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y)
    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(target, P, A, v; regularization)
    @test constrained_optimal_value(target, P, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    AplusE, lambda2 = constrained_AplusE(target, P, A, v; regularization)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v; regularization) ≈ 
            NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y; regularization)
    

    # Hurwitz
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n)) - 2I
    P = ComplexSparsePerturbation(A.!=0)
    v = normalize(rand(ComplexF64, n))
    target = NonHurwitz()
    E, lambda = constrained_minimizer(target, P, A, v)
    @test (A+E)*v ≈ v*lambda
    @test real(lambda) >= 0
    @test constrained_optimal_value(target, P, A, v) ≈ norm(E)^2
    AplusE, lambda2 = constrained_AplusE(target, P, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y)

    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(target, P, A, v; regularization)
    @test real(lambda) >= 0
    @test constrained_optimal_value(target, P, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    AplusE, lambda2 = constrained_AplusE(target, P, A, v; regularization)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v; regularization) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y; regularization) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y; regularization)

    # test sparse

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I
    P = ComplexSparsePerturbation(A.!=0)

    target = NonHurwitz()
    E, lambda = constrained_minimizer(target, P, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))
    @test constrained_optimal_value(target, P, A, v) ≈ norm(E)^2
    AplusE, lambda2 = constrained_AplusE(target, P, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y)

    # test Disc

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A * 0.9 /  maximum(abs.(eigvals(A)))
    P = ComplexSparsePerturbation(A.!=0)

    target = NonSchur()
    E, lambda = constrained_minimizer(target, P, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(A+E))) ≈ 1.
    @test constrained_optimal_value(target, P, A, v) ≈ norm(E)^2
    AplusE, lambda2 = constrained_AplusE(target, P, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y)

    # test sparse

    Random.seed!(0)
    n = 4
    A = sprandn(ComplexF64, n,n, 0.5) / 2
    P = ComplexSparsePerturbation(A.!=0)
    v = normalize(rand(ComplexF64, n))
    target = NonSchur()
    E, lambda = constrained_minimizer(target, P, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(Array(A+E)))) ≈ 1.
    @test constrained_optimal_value(target, P, A, v) ≈ norm(E)^2
    AplusE, lambda2 = constrained_AplusE(target, P, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, P, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, P, A, v, y) ≈ 
    NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P, A, v, y)
end

@testset "General perturbations" begin
      A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
      A = A - 30 * I
      n = size(A, 1)
      target = NonHurwitz()
      v = normalize(rand(ComplexF64, n))
      
      P1 = ComplexSparsePerturbation(A.!=0)
      P2 = GeneralPerturbation(P1)
      
      @test constrained_optimal_value(target, P1, A, v) ≈ constrained_optimal_value(target, P2, A, v)
      E1, lambda1 = constrained_minimizer(target, P1, A, v)
      E2, lambda2 = constrained_minimizer(target, P2, A, v)
      @test E1 ≈ E2
      @test lambda1 ≈ lambda2

      @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P1, A, v) ≈ 
            NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P2, A, v)

      regularization = 0.1
      @test constrained_optimal_value(target, P1, A, v; regularization) ≈ constrained_optimal_value(target, P2, A, v; regularization)
      E1, lambda1 = constrained_minimizer(target, P1, A, v; regularization)
      E2, lambda2 = constrained_minimizer(target, P2, A, v; regularization)
      @test E1 ≈ E2
      @test lambda1 ≈ lambda2
      @test constrained_optimal_value(target, P2, A, v; regularization) ≈ norm(E2)^2 + norm((A+E2)*v-v*lambda2)^2/regularization

      @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P1, A, v; regularization) ≈ 
            NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, P2, A, v; regularization)

      y = randn(ComplexF64, n)
      @test NearestUnstableMatrix.reduced_augmented_Lagrangian(target, P1, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.reduced_augmented_Lagrangian(target, P2, A, v, y; regularization)

      @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P1, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, P2, A, v, y; regularization)

end

function is_toeplitz(A)
      n = size(A, 1)
      return all(allequal(diag(A, k)) for k in -(n-2):(n-2))
end

@testset "Toeplitz perturbation" begin
      A = -[5 4 3.; 2 5 4; 1 2 5]
      P = toeplitz_perturbation(A)
      target = NonHurwitz()
      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))
      x = nearest_unstable(target, P, A, x0,
            stopping_criterion=StopWhenAny(StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))))
      E, lambda = constrained_minimizer(target, P, A, x)
      @test is_toeplitz(E)
      @test abs(lambda - x'*(A+E)*x) < sqrt(eps(1.))
      @test norm((A+E)*x - x*lambda) < sqrt(eps(1.))
      @test abs(real(lambda)) < sqrt(eps(1.))
      fval = constrained_optimal_value(target, P, A, x)
      @assert fval ≈ norm(E)^2

      regularization = 0.1
      E, lambda = constrained_minimizer(target, P, A, x; regularization)
      @test is_toeplitz(E)
      @test abs(real(lambda)) < sqrt(eps(1.))
      fval = constrained_optimal_value(target, P, A, x; regularization)
      @assert fval ≈ norm(E)^2 + norm((A+E)*x-x*lambda)^2/regularization
end

# @testset "Grcar" begin
#       A = -grcar(6)
#       P = toeplitz_perturbation(A, -1:3)
#       target = NonHurwitz()
#       x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))
#       x = nearest_unstable(target, P, A, x0,
#             stopping_criterion=StopWhenAny(StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))))
#       E, lambda = constrained_minimizer(target, P, A, x)
# end

@testset "nearest_unstable" begin
      A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
      A = A - 30 * I
      P = ComplexSparsePerturbation(A.!=0)

      target = Singular() # nearest singular matrix
      # target = Hurwitz # nearest non-Hurwitz stable matrix

      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))

      x = nearest_unstable(target, P, A, x0,
                  stopping_criterion=StopWhenAny(StopAfterIteration(1000), 
                                          StopWhenGradientNormLess(10^(-6))))
      @test constrained_optimal_value(target, P, A, x) ≈ 2.2810193

      x = NearestUnstableMatrix.nearest_unstable_augmented_Lagrangian_method_optim(target, P, A, x0, 
      starting_regularization=3., 
      outer_iterations=30, 
      regularization_damping=0.7,
      gradient=NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic, 
      verbose=false,
      # Optim.jl options
      g_tol=1e-6, 
      iterations=10_000)

      @test constrained_optimal_value(target, P, A, x) ≈ 2.2810193

end
