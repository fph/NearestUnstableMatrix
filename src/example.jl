using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
A = A - 30 * I
pert = ComplexSparsePerturbation(A.!=0)

target = Singular() # nearest singular matrix
# target = Hurwitz # nearest non-Hurwitz stable matrix

x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))

x = nearest_unstable(target, pert, A, x0,
    optimizer=quasi_Newton,
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), 
            (:Cost, " F(x): %1.11f | "), 
            (:GradientNorm, " ||∇F(x)||: %1.11f | "),  
            "\n", :Stop], 
            stopping_criterion=StopWhenAny(StopAfterIteration(1000), 
                                    StopWhenGradientNormLess(10^(-6))))

E, lambda = minimizer(target, pert, A, x)
