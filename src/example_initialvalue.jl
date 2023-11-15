using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

using MatrixMarket
A = mmread("orani678.mtx")
pert = ComplexSparsePerturbation(A.!=0)

target = Singular()

# As initial value, use a full SVD for simplicity 
# (this will not scale well)
U, S, V = svd(Array(A))
x0 = complex.(V[:, end])

x = nearest_unstable(target, pert, A, x0,
#    optimizer=quasi_Newton,
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), 
            (:Cost, " F(x): %1.11f | "), 
            (:GradientNorm, " ||∇F(x)||: %1.11f | "),  
            "\n", :Stop], 
            stopping_criterion=StopWhenAny(StopAfterIteration(1000), 
                                    StopWhenGradientNormLess(10^(-6))))

E, lambda = constrained_minimizer(target, pert, A, x)
