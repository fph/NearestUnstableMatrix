using NearestUnstableMatrix
using Test
using Random
using LinearAlgebra
using SparseArrays

using NearestUnstableMatrix: Schur, project_outside

@testset "project_outside" begin
    @test project_outside(Disc(2.3), 1) == 2.3
    @test project_outside(Disc(2.3), 5) == 5
    @test project_outside(Disc(2.3), 5im) == 5im
    @test project_outside(LeftHalfPlane(2.3), 5im) == 2.3 + 5im
    @test project_outside(LeftHalfPlane(2.3), 5+5im) == 5 + 5im
end


@testset "NearestUnstableMatrix.jl" begin
    A = [1. 2; 3 4]
    v = [3/5;4/5]
    w = [1;0.]
    E = constrained_minimizer(A, v, w)
    @test E ≈ [-0.72 -0.96; -3 -4]
    @test constrained_optimal_value(A, v, w) ≈ norm(E)^2
    
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n))
    v = normalize(rand(ComplexF64, n))
    w = rand(ComplexF64, n)
    E = constrained_minimizer(A, v, w)
    @test (A+E)*v ≈ w
    @test constrained_optimal_value(A, v, w) ≈ norm(E)^2

    # test LHP

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I

    E = constrained_minimizer(A, v, Hurwitz)
    @test (E.==0) == (A.==0)
    lambda = v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))
    @test constrained_optimal_value(A, v, Hurwitz) ≈ norm(E)^2

    # test Disc

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A * 0.9 /  maximum(abs.(eigvals(A)))

    E = constrained_minimizer(A, v, Schur)
    @test (E.==0) == (A.==0)
    lambda = v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(A+E))) ≈ 1.
    @test constrained_optimal_value(A, v, Schur) ≈ norm(E)^2

    # test sparse

    Random.seed!(0)
    n = 4
    A = sprandn(ComplexF64, n,n, 0.5)
    v = normalize(rand(ComplexF64, n))
    w = rand(ComplexF64, n)
    E = constrained_minimizer(A, v, w)
    @test (E.==0) == (A.==0)
    @test (A+E)*v ≈ w
end

