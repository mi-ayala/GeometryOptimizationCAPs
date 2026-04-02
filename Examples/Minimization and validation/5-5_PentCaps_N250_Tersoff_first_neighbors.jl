using GeometryOptimizationCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic
using Parameters
using JLD2

using Distances, LinearAlgebra

### We start by minimizing the harmonic energy. Before using Tersoff potential.
b = 1.4
θ = 2π / 3
kb = 1
kθ = 1

p = Tersoff_parameters()

connectivity, x_initial = get_5_5_connectivity_odd(21)
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)

### BFGS
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, x_initial, method=algo, g_tol=1e-6; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

### Newton refinement
F = x -> extended_Grad(x, x_BFGS, connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_BFGS, connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-13, maxiter=10)[1]

### BFGS optimization stage and Newton refinement 
e = x -> Tersoff_energy(x, p, connectivity)
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, center_nanotube_armchair(x_newton[7:end]), method=algo; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

F = x -> extended_Grad_Tersoff(x, x_BFGS, p, connectivity)
DF = x -> extended_Hess_Tersoff(x, x_BFGS, p, connectivity)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-12, maxiter=3)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]
x_tube = reshape(x_newton[7:end], :, 3)

p_full = (N=250, R=1.95, D=0.15, a=1393.6, b=346.74, λ₁=3.4879, λ₂=2.2119, β=0.00000015724, n=0.72751, h=-0.57058, c=38049.0, d=4.3484)

p_full_interval = (N=250, R=interval(1.95), D=interval(0.15), a=interval(1393.6), b=interval(346.74), λ₁=interval(3.4879), λ₂=interval(2.2119), β=interval(0.00000015724), n=interval(0.72751), h=interval(-0.57058), c=interval(38049.0), d=interval(4.3484))


@inline function cutoff_tag(r::Interval, R::Real, D::Real)
    L = Interval(R - D)  # interval point
    U = Interval(R + D)

    # definitely left / right using strict interval order
    if strictprecedes(r, L)
        return :one
    elseif strictprecedes(U, r)
        return :zero
    else
        # check containment r ⊆ [L,U] via endpoints (floats), no interval comparisons
        r_lo = inf(r)
        r_hi = sup(r)
        Llo = inf(L)
        Uhi = sup(U)

        if (r_lo >= Llo) && (r_hi <= Uhi)
            return :middle
        else
            return :mixed
        end
    end
end

@inline function cutoff_weight(r::Interval, R::Real, D::Real)
    tag = cutoff_tag(r, R, D)

    oneT = one(r)
    zeroT = zero(r)
    twoT = oneT + oneT
    halfT = oneT / twoT

    if tag === :one
        return oneT
    elseif tag === :zero
        return zeroT
    elseif tag === :middle
        πT = Interval(π)
        return halfT * (oneT - sin(πT * (r - R) / (twoT * D)))
    else
        return interval(zeroT, oneT) # [0,1]
    end
end

@inline function dist_row3_interval(X, i::Int, j::Int)
    dx = X[i, 1] - X[j, 1]
    dy = X[i, 2] - X[j, 2]
    dz = X[i, 3] - X[j, 3]
    return sqrt(dx * dx + dy * dy + dz * dz)
end

function build_neighbors_weights_interval!(nbrs::Vector{Vector{Int}},
    w::Vector{Vector{Interval}},
    X::AbstractMatrix{<:Interval},
    p)
    @unpack N, R, D = p
    @assert size(X, 1) == N && size(X, 2) == 3

    @inbounds for i in 1:N
        empty!(nbrs[i])
        empty!(w[i])
        for j in 1:N
            (j == i) && continue
            rij = dist_row3_interval(X, i, j)

            tag = cutoff_tag(rij, R, D)
            tag === :zero && continue  # definitely outside

            push!(nbrs[i], j)
            push!(w[i], cutoff_weight(rij, R, D))
        end
    end
    return nbrs, w
end

function Tersoff_precomputed_weights(X::AbstractMatrix{},
    p,
    nbrs::Vector{Vector{Int}},
    w::Vector{Vector{Interval}})
    @unpack N, a, b, λ₁, λ₂, β, n, h, c, d = p
    @assert size(X, 1) == N && size(X, 2) == 3

    T = eltype(X)     # Interval{Float64}
    oneT = one(T)
    zeroT = zero(T)
    twoT = oneT + oneT
    halfT = oneT / twoT

    # promote parameters to T
    aT = T(a)
    bT = T(b)
    λ1T = T(λ₁)
    λ2T = T(λ₂)
    βT = T(β)
    nT = T(n)
    hT = T(h)
    cT = T(c)
    dT = T(d)

    cd = cT / dT
    cd2 = cd * cd
    c2 = cT * cT
    d2 = dT * dT
    inv2n = -oneT / (twoT * nT)   # -(1)/(2n)

    e = zeroT

    @inbounds for i in 1:N
        Xi1 = X[i, 1]
        Xi2 = X[i, 2]
        Xi3 = X[i, 3]
        nbi = nbrs[i]
        wi = w[i]

        for jj in eachindex(nbi)
            j = nbi[jj]
            fcij = wi[jj]   # precomputed cutoff weight for (i,j)

            dxij = Xi1 - X[j, 1]
            dyij = Xi2 - X[j, 2]
            dzij = Xi3 - X[j, 3]
            rij = sqrt(dxij * dxij + dyij * dyij + dzij * dzij)

            B = zeroT

            for kk in eachindex(nbi)
                k = nbi[kk]
                (k == j) && continue
                fcik = wi[kk]  # cutoff weight for (i,k)

                dxik = Xi1 - X[k, 1]
                dyik = Xi2 - X[k, 2]
                dzik = Xi3 - X[k, 3]
                rik = sqrt(dxik * dxik + dyik * dyik + dzik * dzik)

                # cos(angle) = proj (avoid acos for intervals)
                proj = (dxik * dxij + dyik * dyij + dzik * dzij) / (rij * rik)

                g = oneT + cd2 - c2 / (d2 + (hT - proj) * (hT - proj))
                B += fcik * g
            end

            bij = (oneT + (βT * B)^nT)^(inv2n)

            e += fcij * (aT * exp(-λ1T * rij) + bij * (-bT) * exp(-λ2T * rij))
        end
    end

    e *= halfT
    return e
end

x₀r_interval = interval.(x_tube, 9.9212e-11; format=:midpoint)

nbrs = [Int[] for _ in 1:p_full.N]
w = [Interval[] for _ in 1:p_full.N]

build_neighbors_weights_interval!(nbrs, w, x₀r_interval, p_full_interval)

e(x₀r_interval)
E = Tersoff_precomputed_weights(x₀r_interval, p_full_interval, nbrs, w)

F = x -> Tersoff_precomputed_weights(x, p_full_interval, nbrs, w)
GF_zone = x -> ForwardDiff.gradient(F, x)
HF_zone = x -> ForwardDiff.hessian(F, x)


function extended_Grad_Full_Tersoff(x_input, x_fix, GF)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing vector field
    Fx = GF(reshape(x, :, 3))

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]

    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]


end

function extended_Hess_Full_Tersoff(x_input, x_fix, GF, HF)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing Fx
    ### Computing vector field
    Fx = GF(reshape(x, :, 3))

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]


    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x


    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]

    x = reshape(x, :, 3)

    ### Prelocating Matrices

    dFx = HF(reshape(x, :, 3))


    ### Extending derivative
    dFx = [zeros(type, 3 * N, 6) dFx]
    dFx = [zeros(type, 6, 3 * N + 6); dFx]

    Adjusts = [zeros(type, N, N) -lambda1*Diagonal(ones(type, N)) lambda3*Diagonal(ones(type, N)); lambda1*Diagonal(ones(type, N)) zeros(type, N, N) -lambda2*Diagonal(ones(type, N)); -lambda3*Diagonal(ones(type, N)) lambda2*Diagonal(ones(type, N)) zeros(type, N, N)]
    Adjusts = [I1x I2x I3x T1 T2 T3 Adjusts]
    Adjusts = [zeros(type, 1, 6) transpose(I1xfix); zeros(type, 1, 6) transpose(I2xfix); zeros(type, 1, 6) transpose(I3xfix); zeros(type, 1, 6) transpose(T1); zeros(type, 1, 6) transpose(T2); zeros(type, 1, 6) transpose(T3); Adjusts]

    dFx = dFx + Adjusts

    return dFx

end

### Validation step using interval arithmetic.
F_int = x -> extended_Grad_Full_Tersoff(x, interval.(x_newton[7:end]), GF_zone)
DF_int = x -> extended_Hess_Full_Tersoff(x, interval.(x_newton[7:end]), GF_zone, HF_zone)


@time r = get_proof(x_newton, F_int, DF_int, 9.9212e-11)

# ┌ Info: success: the root and R delimit an interval of existence
# │ Y  = [0.0, 4.2968e-11]_com_NG
# │ Z₁ = [4.88498e-15, 0.000113905]_com_NG
# │ R  = 9.9212e-11
# └ root = [0.0, 4.29729e-11]_com_NG
# 1399.498389 seconds (164.64 M allocations: 541.179 GiB, 1.84% gc time, 0.41% compilation time: 6% of which was recompilation)
# [4.29728e-11, 9.92121e-11]_com_NG
