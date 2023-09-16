using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

using MatrixMarket
A = mmread("orani678.mtx")

target = Nonsingular # nearest singular matrix
# target = Hurwitz # nearest non-Hurwitz stable matrix

n = size(A,1)

M = Manifolds.Sphere(n-1, ℂ)

f(M, v) = constrained_optimal_value(A, v, target; regularization=1e-10)

function g(M, v)
    gr = realgradient(x -> f(M, x), v)
    return project(M, v, gr)
end

# const tape = make_tape(x -> f(M, x), x0)
# function g_rev(M, v)
#     gr = realgradient_reverse(v, tape)
#     return project(M, v, gr)
# end

function g_zygote(M, v)
    gr = first(realgradient_zygote(x -> f(M, x), v))
    return project(M, v, gr)
end

# As initial value, use a full SVD for simplicity 
# (this will not scale well)
U, S, V = svd(Array(A))
x0 = V[:, end]

x = quasi_Newton(M, f, g_zygote, x0; 
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), 
            (:Cost, " F(x): %1.11f | "), 
            (:GradientNorm, " ||∇F(x)||: %1.11f | "),  
            "\n", :Stop], 
            stopping_criterion=StopWhenAny(StopAfterIteration(10000), 
                                    StopWhenGradientNormLess(10^(-6))))

E = constrained_minimizer(A, x, target)
