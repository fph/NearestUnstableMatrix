using LinearAlgebra
using Manifolds
using Optim: optimize, LBFGS
using Optim
using NearestUnstableMatrix

A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
A = A - 30 * I
target = Singular()
pert = ComplexSparsePerturbation(A.!=0)
x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))

x = NearestUnstableMatrix.nearest_unstable_augmented_Lagrangian_method_optim(target, pert, A, x0, 
                        starting_regularization=3., 
                        outer_iterations=30, 
                        regularization_damping=0.7,
                        gradient=NearestUnstableMatrix.reduced_augmented_Lagrangian_Euclidean_gradient_analytic, 
                        # Optim.jl options
                        g_tol=1e-6, 
                        iterations=10_000, 
                        show_trace=true, 
                        show_every=500)

E, lambda = constrained_minimizer(target, pert, A, x)
