using LinearAlgebra
using Manifolds
using Optim: optimize, LBFGS
using Optim
using NearestUnstableMatrix

A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
A = A - 30 * I
target = Singular()

n = size(A,1)

f(v) = constrained_optimal_value(target, A, v)

function g(v)
    gr = realgradient(f, v)
    return gr
end

function g_zygote(v)
    gr = first(realgradient_zygote(f, v))
    return gr
end

x0 = randn(ComplexF64, n)

d = similar(x0)
d .= x0
P = Optim.InverseDiagonal(d)
function precondprep!(P, x)
    P.diag .= x
end

res = optimize(f, g_zygote, x0, Optim.LBFGS(manifold=Optim.Sphere(), P=P, precondprep=precondprep!), Optim.Options(show_trace=true, iterations=10_000), inplace=false)

x = Optim.minimizer(res)

E, lambda = constrained_minimizer(target, A, x)
