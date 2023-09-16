using Manifolds, Manopt, LinearAlgebra, SparseArrays, Random
using NearestUnstableMatrix

using MatrixMarket
A = mmread("orani678.mtx")
n = size(A, 1)

# Random.seed!(0)
# A = sprandn(1000, 1000, 0.05)
# n = size(A,1)

M = Manifolds.Sphere(n-1, ℂ)

# # As initial value, use a full eigendecomposition for simplicity 
# # (this will not scale well)
# eig = eigen(Array(A))
# abslambda, pos = findmin(abs.(eig.values))
# # lambda = eig.values[pos]
# x0 = eig.vectors[:, pos]
# # x0 = project(M, randn(ComplexF64, n))

# As initial value, use a full SVD for simplicity 
# (this will not scale well)
U, S, V = svd(Array(A))
x0 = complex.(V[:, end])
@info "Computed SVD; minimum singular value $(S[end,end])"
target = Nonsingular

for k = 0:10
    @info "Real o.v.: $(constrained_optimal_value(A, x0, target))"

    f(M, v) = constrained_optimal_value(A, v, target; regularization=(10.)^(-k))

    function g(M, v)
        gr = realgradient(x -> f(M, x), v)
        return project(M, v, gr)
    end

    function g_zygote(M, v)
        gr = first(realgradient_zygote(x -> f(M, x), v))
        return project(M, v, gr)
    end

    max_iter = k==10 ? 1000 : 10_000
    x = trust_regions(M, f, g_zygote, x0; 
    debug=[(:Iteration, "$k/%d"),(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), (:GradientNorm, " ||∇F(x)||: %1.11f | "), 1, "\n", :Stop],
    stopping_criterion=StopWhenAny(StopAfterIteration(max_iter), 
    StopWhenGradientNormLess(10^(-6))))
    
    x0 .= x
    @show k
end

E = constrained_minimizer(A, x, target)
