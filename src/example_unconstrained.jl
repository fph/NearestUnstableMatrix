using LinearAlgebra
using Manifolds
using Optim: optimize, LBFGS
using Optim
using NearestUnstableMatrix

A = reshape(collect(1:16), (4,4)); A[1,3:4] .= 0; A[2,4] = 0; A[3,1] = 0; A[4, 1:2] .= 0; A = Float64.(A)
A = A - 30 * I

n = size(A,1)

target = Nonsingular # nearest singular matrix
# target = Hurwitz # nearest non-Hurwitz stable matrix

L = 10_000

f(v) = constrained_optimal_value(A, v, target) + L * (norm(v)-1)^2

function g(v)
    gr = complexgradient(f, v)
    return gr
end

# const tape = make_tape(x -> f(M, x), x0)
# function g_rev(M, v)
#     gr = complexgradient_reverse(v, tape)
#     return project(M, v, gr)
# end

function g_zygote(v)
    gr = first(complexgradient_zygote(f, v))
    return gr
end

x0 = randn(ComplexF64, n)

res = optimize(f, g_zygote, x0, LBFGS(), Optim.Options(show_trace=true, iterations=10_000), inplace=false)

x = Optim.minimizer(res)
E = constrained_minimizer(A, x, target)
