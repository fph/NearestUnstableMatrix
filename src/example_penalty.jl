using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

# A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
# A = A - 30 * I

A = [0 1. 0; 2. 0 0; 0 0 0.5];

target = Nonsingular # nearest singular matrix
# target = Hurwitz # nearest non-Hurwitz stable matrix

n = size(A,1)

M = Manifolds.Sphere(n-1, ℂ)
x0 = project(M, randn(ComplexF64, n))

for regularization = 10. .^ (-1:-1:-12)
    global x0, x
    f(M, v) = constrained_optimal_value(A, v, target; regularization) 

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

    x = quasi_Newton(M, f, g_zygote, x0; debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), (:GradientNorm, " ||∇F(x)||: %1.11f | "),  "\n", :Stop],)
    @show regularization
#    @show x
    x0 = x
end

E = constrained_minimizer(A, x, target)
