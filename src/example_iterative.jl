using Manifolds, Manopt, LinearAlgebra, SparseArrays, Random
using NearestUnstableMatrix

Random.seed!(0)
A = sprandn(400, 400, 0.05)
n = size(A,1)

M = Sphere(n-1, ℂ)

# use a full eigendecomposition for simplicity (will not scale well)
eig = eigen(Array(A))
abslambda, pos = findmin(abs.(eig.values))
lambda = eig.values[pos]
x0 = eig.vectors[:, pos]
# x0 = project(M, randn(ComplexF64, n))

N = 20
for k = 1:N

    global target = OutsideDisc((N-k)/N*abslambda)

    f(M, v) = constrained_optimal_value(A, v, target)

    function g(M, v)
        gr = complexgradient(x -> f(M, x), v)
        return project(M, v, gr)
    end

    # const tape = make_tape(x -> f(M, x), x0)
    # function g_rev(M, v)
    #     gr = complexgradient_reverse(v, tape)
    #     return project(M, v, gr)
    # end

    function g_zygote(M, v)
        gr = first(complexgradient_zygote(x -> f(M, x), v))
        return project(M, v, gr)
    end

    global x = trust_regions(M, f, g_zygote, x0; debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), (:GradientNorm, " ||∇F(x)||: %1.11f | "),  "\n", :Stop],)
    global x0 = x
    @show k
end

E = constrained_minimizer(A, x, target)
