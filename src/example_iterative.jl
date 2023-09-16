using Manifolds, Manopt, LinearAlgebra, SparseArrays, Random
using NearestUnstableMatrix

using MatrixMarket
A = mmread("orani678.mtx")
n = size(A, 1)

# Random.seed!(0)
# A = sprandn(400, 400, 0.05)
# n = size(A,1)

M = Manifolds.Sphere(n-1, ℂ)

# As initial value, use a full eigendecomposition for simplicity 
# (this will not scale well)
eig = eigen(Array(A))
abslambda, pos = findmin(abs.(eig.values))
# lambda = eig.values[pos]
x0 = eig.vectors[:, pos]
# x0 = project(M, randn(ComplexF64, n))

N = 20

for k = 1:N

    global target = OutsideDisc((N-k)/N*abslambda)

    f(M, v) = constrained_optimal_value(A, v, target; regularization=1e-10)

    function g(M, v)
        gr = realgradient(x -> f(M, x), v)
        return project(M, v, gr)
    end

    function g_zygote(M, v)
        gr = first(realgradient_zygote(x -> f(M, x), v))
        return project(M, v, gr)
    end
    max_iter = k<n ? 1000 : 10000

    global x = quasi_Newton(M, f, g_zygote, x0; 
    debug=[(:Iteration, "$k/%d"),(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), (:GradientNorm, " ||∇F(x)||: %1.11f | "), 10, "\n", :Stop],
    stopping_criterion=StopWhenAny(StopAfterIteration(max_iter), 
    StopWhenGradientNormLess(10^(-6))))
    
    global x0 = x
    @show k
end

E = constrained_minimizer(A, x, target)
