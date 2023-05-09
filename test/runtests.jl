using NearestUnstableMatrix
using Test
using Random
using LinearAlgebra

@testset "NearestUnstableMatrix.jl" begin
    # Write your tests here.
    
    
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n))
    v = normalize(rand(ComplexF64, n))
    w = rand(ComplexF64, n)
    E = constrained_minimizer(A, v, w)
    @test (A+E)*v ≈ w
    @test constrained_optimal_value(A, v, w) ≈ norm(E)^2

    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I

    E = constrained_minimizer(A, v)
    @test (E.==0) == (A.==0)
    lambda = v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))

end

