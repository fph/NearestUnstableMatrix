using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

n = 3
EE = fill(zeros(n,n), 2n-1)

EE[1] = [0 0 1; 0 0 0; 0 0 0]
EE[2] = 1/sqrt(2)* [0 1 0; 0 0 1; 0 0 0]
EE[3] = 1/sqrt(3) * Matrix(I, 3,3)
EE[4] = EE[2]'
EE[5] = EE[1]'

p = length(EE)

EEM = reduce(hcat, vec(Ei) for Ei in EE)

v = [1;2;3.]

Mv = reduce(hcat, Ei*v for Ei in EE)

A = [7. 5 2; 6 7 5; 3 6 7]

alpha = EEM' * vec(A)

@assert EEM*alpha ≈ vec(A)
@assert Mv * alpha ≈ A*v

lambda = 0.3

z = v*lambda - A*v

omega = Mv'*inv(Mv*Mv')*z

E = reduce(+, E*m for (E,m) in zip(EE, omega))

@assert EEM'*EEM ≈ I

pc = svd(Mv, full=true)

@assert pc.V[:,1:n]*(Diagonal(pc.S)\ (pc.U'*z)) ≈ omega

@assert EEM * (alpha+omega) ≈ vec(A+E)
v1 = pc.V[:,1:n] * Diagonal(1 ./ pc.S)  * (pc.U' * v*lambda)
v2 = pc.V*Diagonal([zeros(n); ones(p-n)])*pc.V'*alpha

@assert v1 + v2 ≈ alpha + omega

regularization = 0.1
# Mv_reg = hcat(Mv, sqrt(regularization)*I)

# pc_reg = svd(Mv)

nv_reg = (Mv*Mv' + regularization*I) \ z
omega_reg = Mv'*nv_reg
nt = Diagonal((pc.S.^2 .+ regularization)) \ (pc.U' * z)

@assert pc.V[:,1:n]*(Diagonal(pc.S ./ (pc.S.^2 .+ regularization)) * (pc.U'*z)) ≈ omega_reg

E_reg = reduce(+, E*m for (E,m) in zip(EE, omega_reg))

@assert EEM * (alpha+omega_reg) ≈ vec(A+E_reg)

v1_reg = pc.V[:,1:n] * Diagonal(pc.S ./ (pc.S.^2 .+ regularization))  * (pc.U' * v*lambda)
@assert v1_reg ≈ Mv'*inv(Mv*Mv'+regularization*I)*v*lambda
v2_reg = pc.V[:,1:n]*Diagonal(regularization ./ (pc.S.^2 .+ regularization))*(pc.V[:,1:n]'*alpha)
v3_reg = pc.V[:, n+1:end] * (pc.V[:, n+1:end]' * alpha)
@assert v2_reg + v3_reg ≈ alpha - Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha
@assert v1_reg + v2_reg + v3_reg ≈ alpha + omega_reg

w2_reg = pc.U * Diagonal(pc.S .* regularization ./ (pc.S.^2 .+ regularization))* pc.V[:, 1:n]' * alpha
w1_reg = -pc.U*Diagonal(regularization ./ (pc.S.^2 .+ regularization))*pc.U'*v*lambda

@assert w1_reg + w2_reg ≈ (A+E_reg)*v - v*lambda

@assert (v*lambda - (A+E_reg)*v) / regularization ≈ pc.U * nt

# @show mv
# @show sum(abs2, mv[[1,2,3]])
# @show sum(abs2, Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha)

pert = NearestUnstableMatrix.GeneralPerturbation(EE)
target = NearestUnstableMatrix.Singular()
x0 = project(Manifolds.Sphere(size(A,1) - 1, ℂ), randn(Complex{eltype(A)}, size(A, 1)))
x = copy(x0); df_lag = NearestUnstableMatrix.nearest_unstable_augmented_Lagrangian_method!(target, pert, A, x,
           outer_iterations=60, regularization_damping = 0.85,
           debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), 
                   (:Cost, " F(x): %1.11f | "), 
                   (:GradientNorm, " ||∇F(x)||: %1.11f | "),  
                   "\n", 500, :Stop], 
                   stopping_criterion=StopWhenAny(StopAfterIteration(30000), 
                                           StopWhenGradientNormLess(10^(-6))))




v = x                                           
Mv = reduce(hcat, Ei*v for Ei in EE)
pc = svd(Mv, full=true)
E_reg, lambda = minimizer(target, pert, A, x)
                                           
v1_reg = pc.V[:,1:n] * Diagonal(pc.S ./ (pc.S.^2 .+ regularization))  * (pc.U' * v*lambda)
@assert v1_reg ≈ Mv'*inv(Mv*Mv'+regularization*I)*v*lambda
v2_reg = pc.V[:,1:n]*Diagonal(regularization ./ (pc.S.^2 .+ regularization))*(pc.V[:,1:n]'*alpha)
v3_reg = pc.V[:, n+1:end] * (pc.V[:, n+1:end]' * alpha)
@assert v2_reg + v3_reg ≈ alpha - Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha
@assert v1_reg + v2_reg + v3_reg ≈ alpha + omega_reg
