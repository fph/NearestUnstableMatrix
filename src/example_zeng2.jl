using Manifolds, Manopt, LinearAlgebra
using NearestUnstableMatrix
using Polynomials
using GenericLinearAlgebra
using DoubleFloats

p = Polynomial(1.0);
q = Polynomial(1.0);
for j = 1:10
    global p, q
    xj = (-1.)^j * j/2;
    p = p * Polynomial([-xj; 1]);
    q = q * Polynomial([-xj+(10.)^(-j); 1]);
end

d = 4

# normalize as in Bini-Boito
p = p / norm(p);
q = q / norm(q);

degp = degree(p);
degq = degree(q);

"""
Convolution matrix with the coefficients of a polynomial of degree k
"""
function Tk(v, k)
    T = zeros(eltype(v), (length(v)+k, k+1))
    for i = 1:k+1
        T[i:i+length(v)-1, i] = v
    end
    return T
end

"""
Sylvester matrix whose singularity implies a gcd of degree d,
scaled so that norm(Delta)_F^2 = norm(p-deltap)^2 + norm(q-deltaq)^2
"""
scaled_Sylvester_matrix(p::Polynomial, q::Polynomial, d) = scaled_Sylvester_matrix(coeffs(p), coeffs(q), d)
function scaled_Sylvester_matrix(p::AbstractVector, q::AbstractVector, d)
    degp = length(p) - 1
    degq = length(q) - 1
    return hcat(1/sqrt(degq-d+1) * Tk(p, degq-d),
                1/sqrt(degp-d+1) * Tk(q, degp-d));
end

function pad(v, len)
    x = copy(v)
    append!(x, zeros(eltype(v), len-length(v)))
end


A = scaled_Sylvester_matrix(p, q, d)
A = Double64.(A)

x = Polynomial(Array{eltype(p)}([0, 1]))
z = Polynomial(Array{eltype(p)}([0]))

EE = vcat(
    [scaled_Sylvester_matrix(pad(coeffs(x^k), degp+1), pad(coeffs(z), degq+1), d) for k = 0:degp],
    [scaled_Sylvester_matrix(pad(coeffs(z), degp+1), pad(coeffs(x^k), degq+1), d) for k = 0:degq],
)
target = Singular()
pert = GeneralPerturbation([eltype(A).(E) for E in EE])

x0 = project(Manifolds.Sphere(size(A,2) - 1, ℝ), randn(eltype(A), size(A,2)))
x = copy(x0)
y = zeros(eltype(A), size(A,1))
df = nearest_unstable_penalty_method!(target, pert, A, x, y, optimizer=trust_regions!,
           outer_iterations=90,
           starting_regularization = 1.,
           debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), 
                   (:Cost, " F(x): %1.11g | "), 
                   (:GradientNorm, " ||∇F(x)||: %1.11g | "),  
                   "\n", :Stop, 50],
                   stopping_criterion=StopWhenAny(StopAfterIteration(20000), 
                                           StopWhenGradientNormLess(10^(-16))))

Delta = minimizer(target, pert, A, x)[1]
norm(Delta)