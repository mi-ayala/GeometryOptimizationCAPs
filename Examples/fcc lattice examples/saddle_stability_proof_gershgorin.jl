using GeometryOptimizationCAPs

using LinearAlgebra
using StaticArrays
using Statistics
using Random
using JLD2

using IntervalArithmetic
using Arblib

using UnPack

file = load("Examples/fcc lattice examples/data_lattice_saddle.jld2")
x_saddle = file["x_saddle"]
p_d = file["p_d"]

p_int = (interval(1.0), interval(1.0), interval(12.0), interval(6.0), 863)

hessE = x -> -h_LJ(x, p_int)

r = 1.4409e-10
x = interval.(x_saddle, r; format=:midpoint)
x_newton = [interval(zeros(6)); reshape(interval(x_saddle), :, 1)]
N = 863

type = eltype(x)


### Translation generators 
T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]


### Rotations generators at x
I1x = [-x[:, 2]; x[:, 1]; zeros(type, N)]
I2x = [zeros(type, N); -x[:, 3]; x[:, 2]]
I3x = [x[:, 3]; zeros(type, N); -x[:, 1]]

### Modified Hessian 
B = hessE(x) + T1 * transpose(T1) + T2 * transpose(T2) + T3 * transpose(T3) + I1x * transpose(I1x) + I2x * transpose(I2x) + I3x * transpose(I3x)



function diag_for_gershgorin(A::ArbMatrix)
    n = size(A, 1)

    Am = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        Am[i, j] = Float64(Arblib.midpoint(A[i, j]))
    end

    Q = eigen(Symmetric(Am)).vectors

    Qarb = ArbMatrix(Q)
    B = inv(Qarb) * A * Qarb

    return B
end

lo(x::Arb) = Arblib.midpoint(x) - Arblib.radius(x)
hi(x::Arb) = Arblib.midpoint(x) + Arblib.radius(x)
hiabs(x::Arb) = abs(Arblib.midpoint(x)) + Arblib.radius(x)
loabs(x::Arb) = max(Arb(0), abs(Arblib.midpoint(x)) - Arblib.radius(x))

function gershgorin_has_negative_disjoint_disc(B::ArbMatrix)
    n = size(B, 1)

    centers = [B[i, i] for i in 1:n]
    radii   = Vector{Arb}(undef, n)

    for i in 1:n
        r = Arb(0)
        for j in 1:n
            j == i && continue
            r += hiabs(B[i, j])
        end
        radii[i] = r
    end

    for i in 1:n
        hi(centers[i] + radii[i]) < 0 || continue

        disjoint = true
        for j in 1:n
            j == i && continue
            if loabs(centers[i] - centers[j]) <= hi(radii[i] + radii[j])
                disjoint = false
                break
            end
        end

        disjoint && return true, i
    end

    return false, 0
end

### Check for negative eigenvalue using Gerschgorin's theorem
B_arb = diag_for_gershgorin( ArbMatrix(B))
hasneg, idx = gershgorin_has_negative_disjoint_disc(B_arb)

println("Has certified negative eigenvalue: ", hasneg)
println("Disc index: ", idx)