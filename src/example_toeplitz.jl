using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix

n = 3
EE = fill(zeros(n,n), 2n-1)

EE[1] = [0 0 1; 0 0 0; 0 0 0]
EE[2] = 1/sqrt(2)* [0 1 0; 0 0 1; 0 0 0]
EE[3] = 1/sqrt(3) * Matrix(I, 3,3)
EE[4] = EE[2]'
EE[5] = EE[1]'

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

pc = qr(Mv')

@assert pc.Q*(pc.R' \ z) ≈ omega

@assert EEM * (alpha+omega) ≈ vec(A+E)
v1 = pc.Q*(pc.R' \ (v*lambda))
v2 = pc.Q[:,n+1:end]*pc.Q[:,n+1:end]' * alpha

@assert v1 + v2 ≈ alpha + omega

regularization = 0.1
Mv_reg = hcat(Mv, sqrt(regularization)*I)

pc_reg = qr(Mv_reg')

omega_reg = Mv'*inv(Mv*Mv' + regularization*I)*z

@assert (pc_reg.Q*(pc_reg.R' \ z))[1:length(omega_reg)] ≈ omega_reg

E_reg = reduce(+, E*m for (E,m) in zip(EE, omega_reg))

@assert EEM * (alpha+omega_reg) ≈ vec(A+E_reg)

v1_reg = [I zeros(5,3)] * (pc_reg.Q*(pc_reg.R' \ (v*lambda)))
@assert v1_reg ≈ Mv'*inv(Mv*Mv'+regularization*I)*v*lambda
v2_reg = [I zeros(5,3)] *(pc_reg.Q[:,n+1:end]*pc_reg.Q[1:length(EE),n+1:end]' * alpha)
@assert v2_reg ≈ alpha - Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha
@assert v1_reg + v2_reg ≈ alpha + omega_reg

mv = pc_reg.Q'*[alpha;zeros(n)]
@assert (pc_reg.Q * mv[1:n])[1:length(EE)] ≈ Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha

@assert (pc_reg.Q * (pc_reg.R' \ v * lambda - mv[1:n]))[1:length(EE)] ≈ Mv'*inv(Mv*Mv'+regularization*I)*(v*lambda - Mv*alpha)

@show norm(Mv'*inv(Mv*Mv'+regularization*I)*(v*lambda - Mv*alpha))
@show norm((pc_reg.Q * (pc_reg.R' \ v * lambda - mv[1:n]))[1:length(EE)])
@show norm(E_reg)

# @show mv
# @show sum(abs2, mv[[1,2,3]])
# @show sum(abs2, Mv'*inv(Mv*Mv'+regularization*I)*Mv*alpha)


mv[1:n] = pc_reg.R' \ v * lambda
@assert alpha+omega_reg ≈ (pc_reg.Q*mv)[1:length(EE)]

