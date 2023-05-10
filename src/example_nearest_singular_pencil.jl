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


# Manifold of nxn matrices with orthonormal columns
M = Oblique(n, n)

V = rand(M)
points = [1.;; 3;; 4]

V_exact = [nullspace(A+B_exact*points[1]) nullspace(A+B_exact*points[2]) nullspace(A+B_exact*points[3])]


singular_pencil_minimizer(V) = -(A*V + B*(V.*points)) / (V .* points)

pinv_singular_pencil_minimizer(V) = -(A*V + B*(V.*points)) * pinv(V .* points; rtol=1e-8)
tik_singular_pencil_minimizer(V) = -(A*V + B*(V.*points)) * V'  / (V*V' + 1e-8*I)

f(V) = norm(tik_singular_pencil_minimizer(V))^2
g(M, V) = project(M, V, first(gradient(f, V)))

# x0 = rand(M)
x0 = project(M, V_exact+1e-3*randn(n,n))

x = trust_regions(M, (M,x) -> f(x), g, x0; debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), "\n", :Stop],)

E = singular_pencil_minimizer(x)

@show svdvals([A B+E])
@show svdvals([A; B+E])
@show svdvals([A B+E zeros(size(A)); zeros(size(A)) A B+E])

@show norm(E)^2
@show norm(pinv_singular_pencil_minimizer(x))^2
