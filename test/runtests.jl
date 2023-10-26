using NearestUnstableMatrix
using Test
using Random
using LinearAlgebra
using SparseArrays

using NearestUnstableMatrix: Schur, project_outside, compute_m2inv

@testset "project_outside" begin
    @test project_outside(Disc(2.3), 1) == 2.3
    @test project_outside(Disc(2.3), 5) == 5
    @test project_outside(Disc(2.3), 5im) == 5im
    @test project_outside(LeftOf(2.3), 5im) == 2.3 + 5im
    @test project_outside(LeftOf(2.3), 5+5im) == 5 + 5im
end

@testset "compute_m2inv" begin
    P = [0 1; 1 0]
    v = [2; 0]
    @test isequal(compute_m2inv(P, v, 1.0), [1.0; 0.2])
    @test isequal(compute_m2inv(P, v, 0.0), [0.0; 0.25])
    
end

@testset "NearestUnstableMatrix.jl" begin

    # Simple Nonsingular() case
    A = [1. 2; 3 4]
    n = size(A, 1)
    v = [3/5;4/5]
    target = Nonsingular()
    E, lambda = constrained_minimizer(A, v, target)
    @test E ≈ [-1.32 -1.76; -3 -4]
    @test lambda == 0
    @test constrained_optimal_value(A, v, target) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target)
    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(A, v, target; regularization)
    @test constrained_optimal_value(A, v, target; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target; regularization) ≈ 
            NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target; regularization) ≈ 
            NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target; regularization)
    

    # Hurwitz
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n)) - 2I
    v = normalize(rand(ComplexF64, n))
    target = Hurwitz()
    E, lambda = constrained_minimizer(A, v, target)
    @test (A+E)*v ≈ v*lambda
    @test real(lambda) >= 0
    @test constrained_optimal_value(A, v, target) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target)

    # with regularization
    regularization = 0.5
    E, lambda = constrained_minimizer(A, v, target; regularization)
    @test real(lambda) >= 0
    @test constrained_optimal_value(A, v, target; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target; regularization) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target; regularization) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target; regularization)

    # test sparse

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I

    E, lambda = constrained_minimizer(A, v, Hurwitz())
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))
    @test constrained_optimal_value(A, v, Hurwitz()) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target)

    # test Disc

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A * 0.9 /  maximum(abs.(eigvals(A)))

    E, lambda = constrained_minimizer(A, v, Schur())
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(A+E))) ≈ 1.
    @test constrained_optimal_value(A, v, Schur()) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target)

    # test sparse

    Random.seed!(0)
    n = 4
    A = sprandn(ComplexF64, n,n, 0.5) / 2
    v = normalize(rand(ComplexF64, n))
    E, lambda = constrained_minimizer(A, v, Schur())
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(Array(A+E)))) ≈ 1.
    @test constrained_optimal_value(A, v, Schur()) ≈ norm(E)^2
    @test NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_zygote(A, v, target) ≈ 
          NearestUnstableMatrix.constrained_optimal_value_Euclidean_gradient_analytic(A, v, target)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_zygote(A, v, y, target) ≈ 
          NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic(A, v, y, target)

end

