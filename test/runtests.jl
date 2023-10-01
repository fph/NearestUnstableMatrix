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
    @test project_outside(LeftOf(2.3), 5im) == 2.3 + 5im
    @test project_outside(LeftOf(2.3), 5+5im) == 5 + 5im
end


@testset "NearestUnstableMatrix.jl" begin
    A = [1. 2; 3 4]
    v = [3/5;4/5]
    target = Nonsingular()
    E, lambda = constrained_minimizer(A, v, target)
    @test E ≈ [-1.32 -1.76; -3 -4]
    @test lambda == 0
    @test constrained_optimal_value(A, v, target) ≈ norm(E)^2
    
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

end

