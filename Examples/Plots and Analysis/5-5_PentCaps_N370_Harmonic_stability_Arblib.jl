#### Stability proof using Julia and Arblib 
using GeometryOptimizationCAPs
using IntervalArithmetic
using Statistics
using JLD2

using LinearAlgebra
using Arblib

using UnPack

### Load the data
data = load("data/5-5_PentCaps_N370_Harmonic.jld2")
x_newton = data["x_newton"]
r = inf(data["r"])
x = reshape(x_newton[7:end], :, 3)


x = interval.(x, r; format=:midpoint)
x_newton = interval.(x_newton, r; format=:midpoint)

connectivity, _ = get_5_5_connectivity_odd(33)
N = size(connectivity, 1)

type = typeof(x[1, 1])
b = 1.44
θ = 2π / 3
kb = 469
kθ = 63


function Hess(x_input, connectivity, b, θ, kb, kθ)

    ### Input handling

    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])


    x = reshape(x_input, :, 1)

    ### Computing Fx
    ### Computing vector field
    # _, Fx = Energy_Grad(reshape(x, :, 3), connectivity, b, θ, kb, kθ)



    # ### Second alternative
    # Fx = reshape(Fx, 3 * N, 1) 

    ### Computing DFx
    vBondPrime = d -> 2 * kb * (d - b)
    vBondPrimePrime = d -> 2 * kb

    x = reshape(x, :, 3)

    ### Prelocating Matrices

    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    for j = 1:N

        for k = 1:3

            i = connectivity[j, k]
            v = x[i, :] - x[j, :]
            dist = sqrt(dot(v, v))

            ### With respect to x
            dxj = (v[1] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist) .* v
            dxi = (v[1] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist) .* v

            Dxx[j, j] = Dxx[j, j] - dxj[1] + vBondPrime(dist) / dist
            Dxx[i, i] = Dxx[i, i] + dxi[1] + vBondPrime(dist) / dist

            Dxy[j, j] = Dxy[j, j] - dxj[2]
            Dxy[i, i] = Dxy[i, i] + dxi[2]
            Dxz[j, j] = Dxz[j, j] - dxj[3]
            Dxz[i, i] = Dxz[i, i] + dxi[3]

            ### Mixed derivatives
            Dxx[j, i] = Dxx[j, i] - (dxi[1] + vBondPrime(dist) / dist)
            Dxx[i, j] = Dxx[i, j] + (dxj[1] - vBondPrime(dist) / dist)
            Dxy[j, i] = Dxy[j, i] - dxi[2]
            Dxy[i, j] = Dxy[i, j] + dxj[2]
            Dxz[j, i] = Dxz[j, i] - dxi[3]
            Dxz[i, j] = Dxz[i, j] + dxj[3]

            ###  With respect to y
            dyj = v .* (v[2] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dyi = v .* (v[2] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dyy[j, j] = Dyy[j, j] - dyj[2] + vBondPrime(dist) / dist
            Dyy[i, i] = Dyy[i, i] + dyi[2] + vBondPrime(dist) / dist
            Dyz[j, j] = Dyz[j, j] - dyj[3]
            Dyz[i, i] = Dyz[i, i] + dyi[3]

            ### Mixed derivatives
            Dyy[j, i] = Dyy[j, i] - (dyi[2] + vBondPrime(dist) / dist)
            Dyy[i, j] = Dyy[i, j] + (dyj[2] - vBondPrime(dist) / dist)
            Dyz[j, i] = Dyz[j, i] - dyi[3]
            Dyz[i, j] = Dyz[i, j] + dyj[3]

            ### with respect to z
            dzj = v .* (v[3] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dzi = v .* (v[3] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dzz[j, j] = Dzz[j, j] - dzj[3] + vBondPrime(dist) / dist
            Dzz[i, i] = Dzz[i, i] + dzi[3] + vBondPrime(dist) / dist

            ### Mixed derivatives
            Dzz[j, i] = Dzz[j, i] - (dzi[3] + vBondPrime(dist) / dist)
            Dzz[i, j] = Dzz[i, j] + (dzj[3] - vBondPrime(dist) / dist)

        end

    end


    DFx = 0.5 * [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]

    ### Prelocating Matrices
    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)

    Dyx = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzx = zeros(type, N, N)
    Dzy = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    ### ANGLE INTERACTION HESSIAN 
    vAnglePrime = a -> 2 * kθ * (a - θ)
    vAnglePrimePrime = a -> 2 * kθ


    for j = 1:N

        for k = 1:3

            ### Neighbour atoms 
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]

            ### These are vi and vk
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]

            ### These are vi and vk
            dv1_0 = [-1, -1, -1]
            dv1_1 = [1, 1, 1]
            dv1_2 = [0, 0, 0]

            dv2_0 = [-1, -1, -1]
            dv2_1 = [0, 0, 0]
            dv2_2 = [1, 1, 1]

            ### Norms squared
            v1v1 = dot(v1, v1)
            v2v2 = dot(v2, v2)

            ### Derivatives of v1v1 v2v2
            dv1v1_0 = -2 * v1
            dv1v1_1 = 2 * v1
            dv1v1_2 = [0, 0, 0]

            dv2v2_0 = -2 * v2
            dv2v2_1 = [0, 0, 0]
            dv2v2_2 = 2 * v2

            ### Norm nrm
            nrm = sqrt(v1v1 * v2v2)

            ### Derivatives of nrm (dnrm_0 = dnrm1 + dnrm2)        
            dnrm_1 = (v2v2 / nrm) * v1
            dnrm_2 = (v1v1 / nrm) * v2
            dnrm_0 = -dnrm_1 - dnrm_2

            ### Angle and proj
            proj = dot(v1, v2) / nrm
            rad = sqrt(1.0 - proj^2)
            rad2 = 1.0 - proj^2

            angle = acos(proj)

            ### Derivatives of proj(dproj_0 = -dv1-dv2 )
            dproj_1 = v2 / nrm - (proj / v1v1) * v1
            dproj_2 = v1 / nrm - (proj / v2v2) * v2
            dproj_0 = -dproj_1 .- dproj_2

            ### u1 and u2
            u1 = v1 / v1v1
            u2 = v2 / v2v2

            ### Derivatives of u1 and u2
            du1_0 = dv1_0 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_0
            du1_1 = dv1_1 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_1
            du1_2 = dv1_2 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_2
            du2_0 = dv2_0 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_0
            du2_1 = dv2_1 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_1
            du2_2 = dv2_2 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_2


            a = [1, 1, 2, 2, 3, 3]
            b = [2, 3, 3, 1, 1, 2]

            ### BUILDING DERIVATIVES FOR dv1 
            ## dv1 wrt Xi  
            dproj_11 = dv2_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v2 - proj .* du1_1 - u1 .* dproj_1
            Dv_11 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_1) / rad2 + dproj_11) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_1) / rad2

            Dxx[idx1, idx1] = Dxx[idx1, idx1] + Dv_11[1]
            Dyy[idx1, idx1] = Dyy[idx1, idx1] + Dv_11[2]
            Dzz[idx1, idx1] = Dzz[idx1, idx1] + Dv_11[3]

            dproj_11mixed = -dnrm_1[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_1[b] * (v1v1^-2)) - u1[a] .* dproj_1[b]
            Dv_11mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_1[b] / rad2) + dproj_11mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_1[b] / rad2)

            Dxy[idx1, idx1] = Dxy[idx1, idx1] + Dv_11mixed[1]
            Dxz[idx1, idx1] = Dxz[idx1, idx1] + Dv_11mixed[2]
            Dyz[idx1, idx1] = Dyz[idx1, idx1] + Dv_11mixed[3]
            Dyx[idx1, idx1] = Dyx[idx1, idx1] + Dv_11mixed[4]
            Dzx[idx1, idx1] = Dzx[idx1, idx1] + Dv_11mixed[5]
            Dzy[idx1, idx1] = Dzy[idx1, idx1] + Dv_11mixed[6]

            ### dv1 wrt Xk
            dproj_12 = dv2_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v2 - proj .* du1_2 - u1 .* dproj_2
            Dv_12 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_2) / rad2 + dproj_12) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_2) / rad2

            Dxx[idx1, idx2] = Dxx[idx1, idx2] + Dv_12[1]
            Dyy[idx1, idx2] = Dyy[idx1, idx2] + Dv_12[2]
            Dzz[idx1, idx2] = Dzz[idx1, idx2] + Dv_12[3]

            dproj_12mixed = -dnrm_2[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_2[b] * (v1v1^-2)) - u1[a] .* dproj_2[b]
            Dv_12mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_2[b] / rad2) + dproj_12mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_2[b] / rad2)

            Dxy[idx1, idx2] = Dxy[idx1, idx2] + Dv_12mixed[1]
            Dxz[idx1, idx2] = Dxz[idx1, idx2] + Dv_12mixed[2]
            Dyz[idx1, idx2] = Dyz[idx1, idx2] + Dv_12mixed[3]
            Dyx[idx1, idx2] = Dyx[idx1, idx2] + Dv_12mixed[4]
            Dzx[idx1, idx2] = Dzx[idx1, idx2] + Dv_12mixed[5]
            Dzy[idx1, idx2] = Dzy[idx1, idx2] + Dv_12mixed[6]

            ### dv1 wrt Xj   
            dproj_10 = dv2_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v2 - proj .* du1_0 - u1 .* dproj_0
            Dv_10 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_10) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[idx1, j] = Dxx[idx1, j] + Dv_10[1]
            Dyy[idx1, j] = Dyy[idx1, j] + Dv_10[2]
            Dzz[idx1, j] = Dzz[idx1, j] + Dv_10[3]

            dproj_10mixed = -dnrm_0[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_0[b] * (v1v1^-2)) - u1[a] .* dproj_0[b]
            Dv_10mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_0[b] / rad2) + dproj_10mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_0[b] / rad2)

            Dxy[idx1, j] = Dxy[idx1, j] + Dv_10mixed[1]
            Dxz[idx1, j] = Dxz[idx1, j] + Dv_10mixed[2]
            Dyz[idx1, j] = Dyz[idx1, j] + Dv_10mixed[3]
            Dyx[idx1, j] = Dyx[idx1, j] + Dv_10mixed[4]
            Dzx[idx1, j] = Dzx[idx1, j] + Dv_10mixed[5]
            Dzy[idx1, j] = Dzy[idx1, j] + Dv_10mixed[6]

            ### BUILDING DERIVATIVES FOR dv2 
            ### dv2 wrt Xi
            dproj_21 = dv1_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v1 - proj .* du2_1 - u2 .* dproj_1
            Dv_21 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_1) / rad2 + dproj_21) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_1) / rad2

            Dxx[idx2, idx1] = Dxx[idx2, idx1] + Dv_21[1]
            Dyy[idx2, idx1] = Dyy[idx2, idx1] + Dv_21[2]
            Dzz[idx2, idx1] = Dzz[idx2, idx1] + Dv_21[3]

            dproj_21mixed = -dnrm_1[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_1[b] * (v2v2^-2)) - u2[a] .* dproj_1[b]
            Dv_21mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_1[b] / rad2) + dproj_21mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_1[b] / rad2)

            Dxy[idx2, idx1] = Dxy[idx2, idx1] + Dv_21mixed[1]
            Dxz[idx2, idx1] = Dxz[idx2, idx1] + Dv_21mixed[2]
            Dyz[idx2, idx1] = Dyz[idx2, idx1] + Dv_21mixed[3]
            Dyx[idx2, idx1] = Dyx[idx2, idx1] + Dv_21mixed[4]
            Dzx[idx2, idx1] = Dzx[idx2, idx1] + Dv_21mixed[5]
            Dzy[idx2, idx1] = Dzy[idx2, idx1] + Dv_21mixed[6]

            ### dv2 wrt Xk
            dproj_22 = dv1_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v1 - proj .* du2_2 - u2 .* dproj_2
            Dv_22 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_2) / rad2 + dproj_22) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_2) / rad2

            Dxx[idx2, idx2] = Dxx[idx2, idx2] + Dv_22[1]
            Dyy[idx2, idx2] = Dyy[idx2, idx2] + Dv_22[2]
            Dzz[idx2, idx2] = Dzz[idx2, idx2] + Dv_22[3]

            dproj_22mixed = -dnrm_2[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_2[b] * (v2v2^-2)) - u2[a] .* dproj_2[b]
            Dv_22mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_2[b] / rad2) + dproj_22mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_2[b] / rad2)

            Dxy[idx2, idx2] = Dxy[idx2, idx2] + Dv_22mixed[1]
            Dxz[idx2, idx2] = Dxz[idx2, idx2] + Dv_22mixed[2]
            Dyz[idx2, idx2] = Dyz[idx2, idx2] + Dv_22mixed[3]
            Dyx[idx2, idx2] = Dyx[idx2, idx2] + Dv_22mixed[4]
            Dzx[idx2, idx2] = Dzx[idx2, idx2] + Dv_22mixed[5]
            Dzy[idx2, idx2] = Dzy[idx2, idx2] + Dv_22mixed[6]

            ### dv2 wrt Xj  
            dproj_20 = dv1_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v1 - proj .* du2_0 - u2 .* dproj_0
            Dv_20 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_20) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2

            Dxx[idx2, j] = Dxx[idx2, j] + Dv_20[1]
            Dyy[idx2, j] = Dyy[idx2, j] + Dv_20[2]
            Dzz[idx2, j] = Dzz[idx2, j] + Dv_20[3]

            dproj_20mixed = -dnrm_0[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_0[b] * (v2v2^-2)) - u2[a] .* dproj_0[b]
            Dv_20mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_0[b] / rad2) + dproj_20mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_0[b] / rad2)

            Dxy[idx2, j] = Dxy[idx2, j] + Dv_20mixed[1]
            Dxz[idx2, j] = Dxz[idx2, j] + Dv_20mixed[2]
            Dyz[idx2, j] = Dyz[idx2, j] + Dv_20mixed[3]
            Dyx[idx2, j] = Dyx[idx2, j] + Dv_20mixed[4]
            Dzx[idx2, j] = Dzx[idx2, j] + Dv_20mixed[5]
            Dzy[idx2, j] = Dzy[idx2, j] + Dv_20mixed[6]


            ### BUILDING DERIVATIVES FOR dv0 
            ### dv0 wrt Xi

            dproj_01 = -dproj_11 - dproj_21
            Dv_01 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_01) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[j, idx1] = Dxx[j, idx1] + Dv_01[1]
            Dyy[j, idx1] = Dyy[j, idx1] + Dv_01[2]
            Dzz[j, idx1] = Dzz[j, idx1] + Dv_01[3]
            Dxy[j, idx1] = Dxy[j, idx1] - Dv_11mixed[1] - Dv_21mixed[1]
            Dxz[j, idx1] = Dxz[j, idx1] - Dv_11mixed[2] - Dv_21mixed[2]
            Dyz[j, idx1] = Dyz[j, idx1] - Dv_11mixed[3] - Dv_21mixed[3]
            Dyx[j, idx1] = Dyx[j, idx1] - Dv_11mixed[4] - Dv_21mixed[4]
            Dzx[j, idx1] = Dzx[j, idx1] - Dv_11mixed[5] - Dv_21mixed[5]
            Dzy[j, idx1] = Dzy[j, idx1] - Dv_11mixed[6] - Dv_21mixed[6]



            ### dv0 wrt Xk
            dproj_02 = -dproj_12 - dproj_22
            Dv_02 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_02) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2


            Dxx[j, idx2] = Dxx[j, idx2] + Dv_02[1]
            Dyy[j, idx2] = Dyy[j, idx2] + Dv_02[2]
            Dzz[j, idx2] = Dzz[j, idx2] + Dv_02[3]
            Dxy[j, idx2] = Dxy[j, idx2] - Dv_12mixed[1] - Dv_22mixed[1]
            Dxz[j, idx2] = Dxz[j, idx2] - Dv_12mixed[2] - Dv_22mixed[2]
            Dyz[j, idx2] = Dyz[j, idx2] - Dv_12mixed[3] - Dv_22mixed[3]
            Dyx[j, idx2] = Dyx[j, idx2] - Dv_12mixed[4] - Dv_22mixed[4]
            Dzx[j, idx2] = Dzx[j, idx2] - Dv_12mixed[5] - Dv_22mixed[5]
            Dzy[j, idx2] = Dzy[j, idx2] - Dv_12mixed[6] - Dv_22mixed[6]

            ### dv0 wrt Xj      
            dproj_00 = -dproj_10 - dproj_20
            Dv_00 = -(vAnglePrime(angle) / rad) * (proj * (dproj_0 .* dproj_0) / rad2 + dproj_00) + vAnglePrimePrime(angle) * (dproj_0 .* dproj_0) / rad2

            Dxx[j, j] += Dv_00[1]
            Dyy[j, j] += Dv_00[2]
            Dzz[j, j] += Dv_00[3]
            Dxy[j, j] += -Dv_10mixed[1] - Dv_20mixed[1]
            Dxz[j, j] += -Dv_10mixed[2] - Dv_20mixed[2]
            Dyz[j, j] += -Dv_10mixed[3] - Dv_20mixed[3]
            Dyx[j, j] += -Dv_10mixed[4] - Dv_20mixed[4]
            Dzx[j, j] += -Dv_10mixed[5] - Dv_20mixed[5]
            Dzy[j, j] += -Dv_10mixed[6] - Dv_20mixed[6]

        end

    end

    dFx = DFx + [Dxx Dxy Dxz; Dyx Dyy Dyz; Dzx Dzy Dzz]



    return dFx

end

### Proof 

DF_int = x -> Hess(x, connectivity, interval(b),
    interval(θ), interval(kb), interval(kθ))

### Translation generators 
T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

### Rotations generators at x
I1x = [-x[:, 2]; x[:, 1]; zeros(type, N)]
I2x = [zeros(type, N); -x[:, 3]; x[:, 2]]
I3x = [x[:, 3]; zeros(type, N); -x[:, 1]]

B = DF_int(x) + T1 * transpose(T1) + T2 * transpose(T2) + T3 * transpose(T3) + I1x * transpose(I1x) + I2x * transpose(I2x) + I3x * transpose(I3x)

function is_spd(A::ArbMatrix; step::Int=10)
    n, m = size(A)
    n == m || error("A must be square, got $(n)×$(m).")

    det_prev = Arb(1)

    for k in 1:n
        B = A[1:k, 1:k]
        e = ArbMatrix(k, 1); e .= Arb(0)
        e[k,1] = Arb(1)

        x = B \ e
        invdiag = x[k,1]
        ok = Arblib.ispositive(invdiag)

        if (k == 1) || (k % step == 0) || !ok || (k == n)
            println("k = $k / $n  (left: $(n-k))   invdiag>0?  $ok")
        end
        ok || return false

        det_k = det_prev / invdiag
        okdet = Arblib.ispositive(det_k)
        if (k == 1) || (k % step == 0) || !okdet || (k == n)
            println("            det_k>0?      $okdet")
        end
        okdet || return false

        det_prev = det_k
    end

    return true
end

is_positive = is_spd(ArbMatrix(B))
println("Is the Hessian positive definite (SPD)? ", is_positive)


