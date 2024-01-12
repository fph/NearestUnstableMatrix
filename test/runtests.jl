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
    pert = ComplexSparsePerturbation([0 1; 1 0])
    v = [2; 0]
    @test isequal(precompute(pert, v, 1.0), [1.0; 0.2])
    @test isequal(precompute(pert, v, 0.0), [0.0; 0.25])
    
end

@testset "Function values and derivatives" begin

    # Simple Singular() case
    A = [1. 2; 3 4]
    n = size(A, 1)
    v = ComplexF64.([3/5;4/5])
    target = Singular()
    pert = ComplexSparsePerturbation(A.!=0)
    E, lambda = minimizer(target, pert, A, v)
    @test E ≈ [-1.32 -1.76; -3 -4]
    @test lambda == 0
    @test optimal_value(target, pert, A, v) ≈ norm(E)^2
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y)
    # with regularization
    regularization = 0.5
    E, lambda = minimizer(target, pert, A, v; regularization)
    @test optimal_value(target, pert, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v; regularization)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v; regularization) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y; regularization)
    

    # Hurwitz
    Random.seed!(0)
    n = 4
    A = rand(ComplexF64, (n,n)) - 2I
    pert = ComplexSparsePerturbation(A.!=0)
    v = normalize(rand(ComplexF64, n))
    target = NonHurwitz()
    E, lambda = minimizer(target, pert, A, v)
    @test (A+E)*v ≈ v*lambda
    @test real(lambda) >= 0
    @test optimal_value(target, pert, A, v) ≈ norm(E)^2
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y)

    # with regularization
    regularization = 0.5
    E, lambda = minimizer(target, pert, A, v; regularization)
    @test real(lambda) >= 0
    @test optimal_value(target, pert, A, v; regularization) ≈ norm(E)^2 + norm((A+E)*v-v*lambda)^2/regularization
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v; regularization)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v; regularization) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v; regularization)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y; regularization) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y; regularization)

    # test sparse

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A - maximum(real(eigvals(A))) * I - 0.1*I
    pert = ComplexSparsePerturbation(A.!=0)

    target = NonHurwitz()
    E, lambda = minimizer(target, pert, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(real(lambda)) < sqrt(eps(1.))
    @test abs(maximum(real(eigvals(A+E)))) < sqrt(eps(1.))
    @test optimal_value(target, pert, A, v) ≈ norm(E)^2
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y)

    @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v) ≈
          NearestUnstableMatrix.gradient_alternative(target, pert, A, v)

    @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y) ≈
          NearestUnstableMatrix.gradient_alternative(target, pert, A, v, y)


    # test Disc

    Random.seed!(0)
    A = Array(Tridiagonal(rand(ComplexF64,n-1), rand(ComplexF64,n), rand(ComplexF64,n-1)))
    A = A * 0.9 /  maximum(abs.(eigvals(A)))
    pert = ComplexSparsePerturbation(A.!=0)

    target = NonSchur()
    E, lambda = minimizer(target, pert, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(A+E))) ≈ 1.
    @test optimal_value(target, pert, A, v) ≈ norm(E)^2
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y)
    E, lambda = NearestUnstableMatrix.minimizer(target, pert, A, v, y)
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v, y)
    @assert lambda ≈ lambda2
    @assert A+E ≈ AplusE

    # test sparse

    Random.seed!(0)
    n = 4
    A = sprandn(ComplexF64, n,n, 0.5) / 2
    pert = ComplexSparsePerturbation(A.!=0)
    v = normalize(rand(ComplexF64, n))
    target = NonSchur()
    E, lambda = minimizer(target, pert, A, v)
    @test (E.==0) == (A.==0)
    @test lambda ≈ v'*(A+E)*v
    @test (A+E)*v ≈ v*lambda
    @test abs(lambda) ≈ 1.
    @test maximum(abs.(eigvals(Array(A+E)))) ≈ 1.
    @test optimal_value(target, pert, A, v) ≈ norm(E)^2
    AplusE, lambda2, nv = minimizer_AplusE(target, pert, A, v)
    @test AplusE ≈ A+E
    @test lambda2 ≈ lambda
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v) ≈ 
          NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v)
    y = randn(ComplexF64, n)
    @test NearestUnstableMatrix.Euclidean_gradient_zygote(target, pert, A, v, y) ≈ 
    NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert, A, v, y)
end

@testset "General perturbations" begin
      A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
      A = A - 30 * I
      n = size(A, 1)
      target = NonHurwitz()
      v = normalize(rand(ComplexF64, n))
      
      pert1 = ComplexSparsePerturbation(A.!=0)
      pert2 = GeneralPerturbation(pert1)
      
      @test optimal_value(target, pert1, A, v) ≈ optimal_value(target, pert2, A, v)
      E1, lambda1 = minimizer(target, pert1, A, v)
      E2, lambda2 = minimizer(target, pert2, A, v)
      @test E1 ≈ E2
      @test lambda1 ≈ lambda2

      @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert1, A, v) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert2, A, v)

      regularization = 0.1
      @test optimal_value(target, pert1, A, v; regularization) ≈ optimal_value(target, pert2, A, v; regularization)
      E1, lambda1 = minimizer(target, pert1, A, v; regularization)
      E2, lambda2 = minimizer(target, pert2, A, v; regularization)
      @test E1 ≈ E2
      @test lambda1 ≈ lambda2
      @test optimal_value(target, pert2, A, v; regularization) ≈ norm(E2)^2 + norm((A+E2)*v-v*lambda2)^2/regularization

      @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert1, A, v; regularization) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert2, A, v; regularization)

      y = randn(ComplexF64, n)
      @test NearestUnstableMatrix.optimal_value(target, pert1, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.optimal_value(target, pert2, A, v, y; regularization)

      @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert1, A, v, y; regularization) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert2, A, v, y; regularization)

end

function is_toeplitz(A)
      n = size(A, 1)
      return all(allequal(diag(A, k)) for k in -(n-2):(n-2))
end

@testset "Toeplitz perturbation" begin
      A = -[5 4 3.; 2 5 4; 1 2 5]
      pert = toeplitz_perturbation(A)
      target = NonHurwitz()
      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))
      x = nearest_unstable(target, pert, A, x0,
            stopping_criterion=StopWhenAny(StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))))
      E, lambda = minimizer(target, pert, A, x)
      @test is_toeplitz(E)
      @test abs(lambda - x'*(A+E)*x) < sqrt(eps(1.))
      @test norm((A+E)*x - x*lambda) < sqrt(eps(1.))
      @test abs(real(lambda)) < sqrt(eps(1.))
      fval = optimal_value(target, pert, A, x)
      @assert fval ≈ norm(E)^2

      regularization = 0.1
      E, lambda = minimizer(target, pert, A, x; regularization)
      @test is_toeplitz(E)
      @test abs(real(lambda)) < sqrt(eps(1.))
      fval = optimal_value(target, pert, A, x; regularization)
      @assert fval ≈ norm(E)^2 + norm((A+E)*x-x*lambda)^2/regularization
end

@testset "Matrix polynomials" begin
      Random.seed!(0)
      target = Singular()
      n = 4
      k = 2
      d = 4
      A = randn(ComplexF64, (n, n, k+1))
      v = randn(ComplexF64, (n, 1, d+1))
      O = zeros(n,n)
      T = [A[:,:,1] O        O        O        O; 
           A[:,:,2] A[:,:,1] O        O        O;
           A[:,:,3] A[:,:,2] A[:,:,1] O        O;
           O        A[:,:,3] A[:,:,2] A[:,:,1] O;
           O        O        A[:,:,3] A[:,:,2] A[:,:,1];
           O        O        O        A[:,:,3] A[:,:,2];
           O        O        O        O        A[:,:,3]]
      @test NearestUnstableMatrix.product(A, v) ≈ T*v[:]
      y = randn(ComplexF64, n*(k+d+1))
      @test NearestUnstableMatrix.adjoint_product(A, y) ≈ T'*y

      pert1 = UnstructuredPerturbation(A)
      pert2 = unstructured_perturbation(A)
      @test optimal_value(target, pert1, A, v, y; regularization=0.1) ≈ optimal_value(target, pert2, A, v, y; regularization=0.1)
      @test NearestUnstableMatrix.optimal_value_naif(target, pert1, A, v, y; regularization=0.1) ≈ 
            optimal_value(target, pert1, A, v, y; regularization=0.1)

      E, lambda = minimizer(target, pert1, A, v)
      @test optimal_value(target, pert1, A, v) ≈ norm(E)^2

      @test NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert1, A, v, y; regularization=0.1) ≈ 
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert2, A, v, y; regularization=0.1)
      
      @test realgradient(x->NearestUnstableMatrix.optimal_value_naif(target, pert1, A, x, y; regularization=0.1), v[:]) ≈
            NearestUnstableMatrix.Euclidean_gradient_analytic(target, pert1, A, v, y; regularization=0.1)

      w = randn(ComplexF64, (n, 1, d+1))
      H = NearestUnstableMatrix.realhessian_zygote(x -> NearestUnstableMatrix.optimal_value_naif(target, pert1, A, x), v[:])
      @test_broken NearestUnstableMatrix.Euclidean_Hessian_product_analytic(w[:], target, pert1, A, v) ≈ H*w
end

@testset "Grcar" begin
      Random.seed!(1)
      A = -grcar(6)
      pert = toeplitz_perturbation(A, -1:3)
      target = NonHurwitz()
      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))
      x = nearest_unstable(target, pert, A, x0, regularization=1e-4,
            stopping_criterion=StopWhenAny(StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))))
      @test optimal_value(target, pert, A, x, regularization=1e-4) ≈ 0.21364254813 # keep checked, we're not 100% sure the method converges to this value for all choices of x0.
end

@testset "nearest_unstable" begin
      A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
      A = A - 30 * I
      pert = ComplexSparsePerturbation(A.!=0)

      target = Singular()

      x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))

      x = nearest_unstable(target, pert, A, x0,
                  stopping_criterion=StopWhenAny(StopAfterIteration(1000), 
                                          StopWhenGradientNormLess(10^(-6))))
      @test optimal_value(target, pert, A, x) ≈ 2.2810193

      x = NearestUnstableMatrix.nearest_unstable_augmented_Lagrangian_method_optim(target, pert, A, x0, 
      starting_regularization=3., 
      outer_iterations=30, 
      regularization_damping=0.7,
      gradient=NearestUnstableMatrix.Euclidean_gradient_analytic, 
      verbose=false,
      # Optim.jl options
      g_tol=1e-6, 
      iterations=10_000)

      @test optimal_value(target, pert, A, x) ≈ 2.2810193

end
