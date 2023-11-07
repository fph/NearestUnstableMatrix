using NearestUnstableMatrix
using Test
using Random
using LinearAlgebra
using SparseArrays
using Manifolds, Manopt

using NearestUnstableMatrix: NonSchur, project, compute_m2inv

@testset "project" begin
    @test project(OutsideDisc(2.3), 1) == 2.3
    @test project(OutsideDisc(2.3), 5) == 5
    @test project(OutsideDisc(2.3), 5im) == 5im
    @test project(RightOf(2.3), 5im) == 2.3 + 5im
    @test project(RightOf(2.3), 5+5im) == 5 + 5im
end

@testset "compute_m2inv" begin
    P = [0 1; 1 0]
    v = [2; 0]
    @test isequal(compute_m2inv(P, v, 1.0), [1.0; 0.2])
    @test isequal(compute_m2inv(P, v, 0.0), [0.0; 0.25])
    
end

@testset "Function values and derivatives" begin

    # Simple Singular() case
    A = [1. 2; 3 4]
    n = size(A, 1)
    v = ComplexF64.([3/5;4/5])
    target = Singular()
    E, lambda = constrained_minimizer(target, A, v)
    @test E ≈ [-1.32 -1.76; -3 -4]
    @test lambda == 0
    @test constrained_optimal_value(target, A, v) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y)
    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(target, A, v; regularization)
    @test constrained_optimal_value(target, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v; regularization) ≈ 
            NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y; regularization)
    

    # Hurwitz
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n)) - 2I
    v = normalize(rand(ComplexF64, n))
    target = NonHurwitz()
    E, lambda = constrained_minimizer(target, A, v)
    @test (A+E)*v ≈ v*lambda
    @test real(lambda) >= 0
    @test constrained_optimal_value(target, A, v) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y)

    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(target, A, v; regularization)
    @test real(lambda) >= 0
    @test constrained_optimal_value(target, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v; regularization) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y; regularization) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y; regularization)

    # test sparse

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I

    target = NonHurwitz()
    E, lambda = constrained_minimizer(target, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))
    @test constrained_optimal_value(target, A, v) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y)

    # test Disc

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A * 0.9 /  maximum(abs.(eigvals(A)))

    target = NonSchur()
    E, lambda = constrained_minimizer(target, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(A+E))) ≈ 1.
    @test constrained_optimal_value(target, A, v) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y)

    # test sparse

    Random.seed!(0)
    n = 4
    A = sprandn(ComplexF64, n,n, 0.5) / 2
    v = normalize(rand(ComplexF64, n))
    target = NonSchur()
    E, lambda = constrained_minimizer(target, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(Array(A+E)))) ≈ 1.
    @test constrained_optimal_value(target, A, v) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(target, A, v) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(target, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(target, A, v, y) ≈ 
    NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(target, A, v, y)
end


@testset "nearest_unstable" begin
      A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
      A = A - 30 * I

      target = Singular() # nearest singular matrix
      # target = Hurwitz # nearest non-Hurwitz stable matrix

      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))

      x = nearest_unstable(target, A, x0,
                  stopping_criterion=StopWhenAny(StopAfterIteration(1000), 
                                          StopWhenGradientNormLess(10^(-6))))
      @test constrained_optimal_value(target, A, x) ≈ 2.2810193

      x = NearestUnstableMatrix.augmented_Lagrangian_method_optim(target, A, x0, 
      starting_regularization=3., 
      outer_iterations=30, 
      regularization_damping=0.7,
      gradient=NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic, 
      # Optim.jl options
      g_tol=1e-6, 
      iterations=10_000)

      @test constrained_optimal_value(target, A, x) ≈ 2.2810193

end

