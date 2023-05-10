using Manifolds, Manopt, LinearAlgebra, Zygote

n = 3
# A = randn(n, n-1) * randn(n-1, n)
# B = randn(n, n)

A = [1. 0 0;
     0  0 0;
     0  0 1]
B_exact = [0 1. 0; 0 0 1; 0 0 0];
mu = 1e-3;
B = copy(B_exact); B[2,2] = mu; B[3,1] = mu

# We look for the closest singular pencil of the form Ax+(B+E).

# Manifold of nx(n-1) matrices with orthonormal columns
M = Euclidean(n)^(n-1)

w0 = nullspace(A)[:,1]

function singular_pencil_minimizer(W)
    mat = [w0 W]
    rhs = -[A*W[:,1]+B*w0 A*W[:,2]+B*W[:,1] B*W[:,2]]
    E = rhs / mat
    return E
end

f(W) = norm(singular_pencil_minimizer(W))^2
g(M, W) = project(M, W, first(gradient(f, W)))

x0 = rand(M)

x = trust_regions(M, (M,x) -> f(x), g, x0; debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), "\n", :Stop],)

E = singular_pencil_minimizer(x)





