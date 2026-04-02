using LinearAlgebra
using StaticArrays
using Random
using Statistics
using ForwardDiff
using Distances
using DifferentialEquations
using NearestNeighbors
using Printf
using JLD2
using RadiiPolynomial
using IntervalArithmetic


function build_fcc_block(σ::Real)
    L = 6 * σ
    n = 4
    a = L / n

    basis = (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )

    N = 4 * n^3
    X = zeros(promote_type(typeof(σ), Float64), N, 3)

    t = 1
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        sx = i * a
        sy = j * a
        sz = k * a

        for b in basis
            X[t, 1] = sx + a * b[1]
            X[t, 2] = sy + a * b[2]
            X[t, 3] = sz + a * b[3]
            t += 1
        end
    end

    return X, L, a
end

wrap_mic(dx, L) = dx - L * round(dx / L)

function g_pbc!(du, u, p, t)
    σ, ϵ, α, λ, N, L, rc = p

    T = eltype(u)
    σT = T(σ)
    ϵT = T(ϵ)
    αT = T(α)
    λT = T(λ)
    LT = T(L)
    rc2 = T(rc)^2

    @views x = u[1:N]
    @views y = u[N+1:2N]
    @views z = u[2N+1:3N]

    @views dux = du[1:N]
    @views duy = du[N+1:2N]
    @views duz = du[2N+1:3N]

    fill!(du, zero(T))

    σ6 = σT^6
    c = 4 * ϵT * σ6
    λp2 = λT + 2
    αp2 = αT + 2
    invN = one(T) / T(N)

    @inbounds for i in 1:N-1
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in i+1:N
            dx = wrap_mic(xi - x[j], LT)
            dy = wrap_mic(yi - y[j], LT)
            dz = wrap_mic(zi - z[j], LT)

            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= rc2
                r = sqrt(r2)

                f = c * (λT * r^(-λp2) - (αT * σ6) * r^(-αp2))

                dux[i] -= f * dx
                duy[i] -= f * dy
                duz[i] -= f * dz

                dux[j] += f * dx
                duy[j] += f * dy
                duz[j] += f * dz
            end
        end
    end

    du .*= -invN
    return nothing
end

g_pbc(u, p) = (du = similar(u); g_pbc!(du, u, p, 1.0); du)

function h_pbc!(Jac, u, p, t)
    σ, ϵ, α, λ, N, L, rc = p

    T = eltype(u)
    σT = T(σ)
    ϵT = T(ϵ)
    αT = T(α)
    λT = T(λ)
    LT = T(L)
    rc2 = T(rc)^2

    @views x = u[1:N]
    @views y = u[N+1:2N]
    @views z = u[2N+1:3N]

    fill!(Jac, zero(eltype(Jac)))

    σ6 = σT^6
    c = 4 * ϵT * σ6
    λp2 = λT + 2
    αp2 = αT + 2
    invN = one(T) / T(N)

    @inbounds for i in 1:N-1
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in i+1:N
            dx = wrap_mic(xi - x[j], LT)
            dy = wrap_mic(yi - y[j], LT)
            dz = wrap_mic(zi - z[j], LT)

            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= rc2
                r = sqrt(r2)

                f = c * (λT * r^(-λp2) - (αT * σ6) * r^(-αp2))
                a = c * (-λT * λp2 * r^(-λp2 - 2) + (αT * αp2 * σ6) * r^(-αp2 - 2))

                Hxx = dx * dx * a + f
                Hyy = dy * dy * a + f
                Hzz = dz * dz * a + f
                Hxy = dx * dy * a
                Hxz = dx * dz * a
                Hyz = dy * dz * a

                ii = i
                jj = j
                iN = N + i
                jN = N + j
                i2N = 2N + i
                j2N = 2N + j

                Jac[ii, ii] += Hxx
                Jac[ii, iN] += Hxy
                Jac[ii, i2N] += Hxz

                Jac[iN, ii] += Hxy
                Jac[iN, iN] += Hyy
                Jac[iN, i2N] += Hyz

                Jac[i2N, ii] += Hxz
                Jac[i2N, iN] += Hyz
                Jac[i2N, i2N] += Hzz

                Jac[jj, jj] += Hxx
                Jac[jj, jN] += Hxy
                Jac[jj, j2N] += Hxz

                Jac[jN, jj] += Hxy
                Jac[jN, jN] += Hyy
                Jac[jN, j2N] += Hyz

                Jac[j2N, jj] += Hxz
                Jac[j2N, jN] += Hyz
                Jac[j2N, j2N] += Hzz

                Jac[ii, jj] -= Hxx
                Jac[ii, jN] -= Hxy
                Jac[ii, j2N] -= Hxz

                Jac[iN, jj] -= Hxy
                Jac[iN, jN] -= Hyy
                Jac[iN, j2N] -= Hyz

                Jac[i2N, jj] -= Hxz
                Jac[i2N, jN] -= Hyz
                Jac[i2N, j2N] -= Hzz

                Jac[jj, ii] -= Hxx
                Jac[jj, iN] -= Hxy
                Jac[jj, i2N] -= Hxz

                Jac[jN, ii] -= Hxy
                Jac[jN, iN] -= Hyy
                Jac[jN, i2N] -= Hyz

                Jac[j2N, ii] -= Hxz
                Jac[j2N, iN] -= Hyz
                Jac[j2N, i2N] -= Hzz
            end
        end
    end

    Jac .*= -invN
    return nothing
end

function h_pbc(u, p)
    _, _, _, _, N, _, _ = p
    Jac = Matrix{eltype(u)}(undef, 3N, 3N)
    h_pbc!(Jac, u, p, 1.0)
    return Jac
end

function pbc_neighbors(X::AbstractMatrix, L::Real; rc::Real=Inf, return_sorted::Bool=true)
    N = size(X, 1)
    @assert size(X, 2) == 3

    T = promote_type(eltype(X), typeof(L), typeof(rc))
    rc2 = T(rc)^2

    nbr_ids = [Int[] for _ in 1:N]
    nbr_dist = [T[] for _ in 1:N]
    nbr_disp = [NTuple{3,T}[] for _ in 1:N]

    @inbounds for i in 1:N-1
        xi, yi, zi = X[i, 1], X[i, 2], X[i, 3]
        for j in i+1:N
            dx = wrap_mic(X[j, 1] - xi, L)
            dy = wrap_mic(X[j, 2] - yi, L)
            dz = wrap_mic(X[j, 3] - zi, L)

            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= rc2
                r = sqrt(r2)

                # store j as neighbor of i
                push!(nbr_ids[i], j)
                push!(nbr_dist[i], r)
                push!(nbr_disp[i], (dx, dy, dz))

                # store i as neighbor of j
                push!(nbr_ids[j], i)
                push!(nbr_dist[j], r)
                push!(nbr_disp[j], (-dx, -dy, -dz))
            end
        end
    end

    if return_sorted
        for i in 1:N
            p = sortperm(nbr_dist[i])
            nbr_ids[i] = nbr_ids[i][p]
            nbr_dist[i] = nbr_dist[i][p]
            nbr_disp[i] = nbr_disp[i][p]
        end
    end

    return nbr_ids, nbr_dist, nbr_disp
end

_T(::Type{T}, x) where {T} = convert(T, x)

function _unique_round_int(x)
    lo = round(Int, inf(x))
    hi = round(Int, sup(x))
    lo == hi || error("minimum-image integer is not unique on interval $x; subdivide/refine the box")
    return lo
end

function mic_scalar(dx, L)
    q = dx / L
    k = _unique_round_int(q)
    return dx - L * interval(k)
end

function pbc_shift_matrices(u0, N, L)
    x = u0[1:N]
    y = u0[N+1:2N]
    z = u0[2N+1:3N]

    Kx = zeros(Int, N, N)
    Ky = zeros(Int, N, N)
    Kz = zeros(Int, N, N)

    @inbounds for i in 1:N-1
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in i+1:N
            kx = round(Int, (xi - x[j]) / L)
            ky = round(Int, (yi - y[j]) / L)
            kz = round(Int, (zi - z[j]) / L)

            Kx[i, j] = kx
            Ky[i, j] = ky
            Kz[i, j] = kz

            Kx[j, i] = -kx
            Ky[j, i] = -ky
            Kz[j, i] = -kz
        end
    end

    return Kx, Ky, Kz
end

function check_pbc_shifts_cutoff(u, N, L, rc, Kx, Ky, Kz)
    x = u[1:N]
    y = u[N+1:2N]
    z = u[2N+1:3N]

    LT = interval(L)
    rc2 = interval(rc)^2
    halfL = LT / interval(2)

    @inbounds for i in 1:N-1
        for j in i+1:N
            dx = (x[i] - x[j]) - LT * interval(Kx[i, j])
            dy = (y[i] - y[j]) - LT * interval(Ky[i, j])
            dz = (z[i] - z[j]) - LT * interval(Kz[i, j])

            r2 = dx * dx + dy * dy + dz * dz

            # if definitely outside cutoff, no need to check image uniqueness
            if inf(r2) > sup(rc2)
                continue
            end

            # only now require the chosen image to stay strictly inside (-L/2, L/2)
            if !(inf(dx) > -sup(halfL) && sup(dx) < sup(halfL))
                error("x-image changed or touches boundary for interacting pair ($i,$j)")
            end
            if !(inf(dy) > -sup(halfL) && sup(dy) < sup(halfL))
                error("y-image changed or touches boundary for interacting pair ($i,$j)")
            end
            if !(inf(dz) > -sup(halfL) && sup(dz) < sup(halfL))
                error("z-image changed or touches boundary for interacting pair ($i,$j)")
            end
        end
    end
    return true
end

function g_LJ_pbc_fixedshifts!(du, u, p, t)
    σ, ϵ, α, λ, N, L, rc, Kx, Ky, Kz = p
    N = Int(N)

    T = eltype(u)

    σT = T(σ)
    ϵT = T(ϵ)
    LT = T(L)
    rcT = T(rc)

    αi = α
    λi = λ
    λp2 = λi + interval(2)
    αp2 = αi + interval(2)

    @views x = u[1:N]
    @views y = u[N+1:2N]
    @views z = u[2N+1:3N]

    @views dux = du[1:N]
    @views duy = du[N+1:2N]
    @views duz = du[2N+1:3N]

    fill!(du, zero(T))

    four = interval(4)
    invN = inv(interval(N))

    σ6 = σT^6
    c = four * ϵT * σ6
    rc2 = rcT^2

    @inbounds for i in 1:N-1
        xi = x[i]
        yi = y[i]
        zi = z[i]

        for j in i+1:N
            dx = (xi - x[j]) - LT * interval(Kx[i, j])
            dy = (yi - y[j]) - LT * interval(Ky[i, j])
            dz = (zi - z[j]) - LT * interval(Kz[i, j])

            r2 = dx * dx + dy * dy + dz * dz

            if sup(r2) <= sup(rc2)
                r = sqrt(r2)

                f = c * (λi * inv(r^λp2) - (αi * σ6) * inv(r^αp2))

                dux[i] -= f * dx
                duy[i] -= f * dy
                duz[i] -= f * dz

                dux[j] += f * dx
                duy[j] += f * dy
                duz[j] += f * dz

            elseif inf(r2) > sup(rc2)
                nothing
            else
                error("pair ($i,$j) intersects the cutoff boundary; shrink/subdivide")
            end
        end
    end

    du .*= -invN
    return nothing
end

g_LJ_pbc_fixedshifts(u, p) = begin
    du = similar(u)
    fill!(du, zero(eltype(u)))
    g_LJ_pbc_fixedshifts!(du, u, p, one(eltype(u)))
    du
end

function h_LJ_pbc_fixedshifts!(Jac, u, p, t)
    σ, ϵ, α, λ, N, L, rc, Kx, Ky, Kz = p
    N = Int(N)

    T = eltype(u)

    σT = T(σ)
    ϵT = T(ϵ)
    LT = T(L)
    rcT = T(rc)

    αi = α
    λi = λ
    αp2 = αi + interval(2)
    λp2 = λi + interval(2)

    @views x = u[1:N]
    @views y = u[N+1:2N]
    @views z = u[2N+1:3N]

    fill!(Jac, zero(T))

    σ6 = σT^6
    c = interval(4) * ϵT * σ6
    rc2 = rcT^2
    invN = inv(interval(N))

    @inbounds for i in 1:N-1
        xi = x[i]
        yi = y[i]
        zi = z[i]

        for j in i+1:N
            dx = (xi - x[j]) - LT * interval(Kx[i, j])
            dy = (yi - y[j]) - LT * interval(Ky[i, j])
            dz = (zi - z[j]) - LT * interval(Kz[i, j])

            r2 = dx * dx + dy * dy + dz * dz

            if sup(r2) <= sup(rc2)
                r = sqrt(r2)

                f = c * (λi * inv(r^λp2) - (αi * σ6) * inv(r^αp2))
                a = c * (
                    -λi * (λi + interval(2)) * inv(r^(λp2 + interval(2))) +
                    +αi * (αi + interval(2)) * σ6 * inv(r^(αp2 + interval(2)))
                )

                Hxx = dx * dx * a + f
                Hyy = dy * dy * a + f
                Hzz = dz * dz * a + f

                Hxy = dx * dy * a
                Hxz = dx * dz * a
                Hyz = dy * dz * a

                ix = i
                jx = j
                iy = N + i
                jy = N + j
                iz = 2N + i
                jz = 2N + j

                Jac[ix, ix] += Hxx
                Jac[ix, iy] += Hxy
                Jac[ix, iz] += Hxz
                Jac[iy, ix] += Hxy
                Jac[iy, iy] += Hyy
                Jac[iy, iz] += Hyz
                Jac[iz, ix] += Hxz
                Jac[iz, iy] += Hyz
                Jac[iz, iz] += Hzz

                Jac[jx, jx] += Hxx
                Jac[jx, jy] += Hxy
                Jac[jx, jz] += Hxz
                Jac[jy, jx] += Hxy
                Jac[jy, jy] += Hyy
                Jac[jy, jz] += Hyz
                Jac[jz, jx] += Hxz
                Jac[jz, jy] += Hyz
                Jac[jz, jz] += Hzz

                Jac[ix, jx] -= Hxx
                Jac[ix, jy] -= Hxy
                Jac[ix, jz] -= Hxz
                Jac[iy, jx] -= Hxy
                Jac[iy, jy] -= Hyy
                Jac[iy, jz] -= Hyz
                Jac[iz, jx] -= Hxz
                Jac[iz, jy] -= Hyz
                Jac[iz, jz] -= Hzz

                Jac[jx, ix] -= Hxx
                Jac[jx, iy] -= Hxy
                Jac[jx, iz] -= Hxz
                Jac[jy, ix] -= Hxy
                Jac[jy, iy] -= Hyy
                Jac[jy, iz] -= Hyz
                Jac[jz, ix] -= Hxz
                Jac[jz, iy] -= Hyz
                Jac[jz, iz] -= Hzz

            elseif inf(r2) > sup(rc2)
                nothing
            else
                error("pair ($i,$j) intersects the cutoff boundary; shrink/subdivide")
            end
        end
    end

    Jac .*= -invN
    return nothing
end

h_LJ_pbc_fixedshifts(u, p) = begin
    N = Int(p[5])
    T = eltype(u)
    Jac = zeros(T, 3N, 3N)
    h_LJ_pbc_fixedshifts!(Jac, u, p, one(T))
    Jac
end

function extended_Grad_pbc(x_input, x_fix, p, gradE)

    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, :, 1)
    T = eltype(x_input[1])

    N = Int(size(x_fix, 1) ÷ 3)


    # lambda1 = x_input[1]
    # lambda2 = x_input[2]
    # lambda3 = x_input[3]
    mu1 = x_input[1]
    mu2 = x_input[2]
    mu3 = x_input[3]

    x = reshape(x_input[4:end], :, 1)

    # vector field 
    Fx = gradE(reshape(x, :, 3))
    Fx = reshape(Fx, 3N, 1)

    #  Translation generators 
    o = one(T)
    z = zero(T)

    T1 = vcat(fill(o, N, 1), fill(z, 2N, 1))
    T2 = vcat(fill(z, N, 1), fill(o, N, 1), fill(z, N, 1))
    T3 = vcat(fill(z, 2N, 1), fill(o, N, 1))


    T1x = (hcat(fill(o, 1, N), fill(z, 1, 2N)) * x)
    T2x = (hcat(fill(z, 1, N), fill(o, 1, N), fill(z, 1, N)) * x)
    T3x = (hcat(fill(z, 1, 2N), fill(o, 1, N)) * x)

    # Rotation generators 
    # I1xfix = vcat(-x_fix[N+1:2N, :], x_fix[1:N, :], fill(z, N, size(x_fix, 2)))
    # I2xfix = vcat(fill(z, N, size(x_fix, 2)), -x_fix[2N+1:end, :], x_fix[N+1:2N, :])
    # I3xfix = vcat(x_fix[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x_fix[1:N, :])

    # I1x = vcat(-x[N+1:2N, :], x[1:N, :], fill(z, N, size(x_fix, 2)))
    # I2x = vcat(fill(z, N, size(x_fix, 2)), -x[2N+1:end, :], x[N+1:2N, :])
    # I3x = vcat(x[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x[1:N, :])


    Fx_ext = Fx + mu1 * T1 + mu2 * T2 + mu3 * T3
    # + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    # balancing equations 
    # bal1 = (transpose(x) * I1xfix)
    # bal2 = (transpose(x) * I2xfix)
    # bal3 = (transpose(x) * I3xfix)

    return vcat(T1x, T2x, T3x, Fx_ext)
end

function extended_Hess_pbc(x_input, x_fix, p, gradE, hessE)

    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, :, 1)
    T = eltype(x_input[1])

    N = Int(size(x_fix, 1) ÷ 3)

    # lambda1 = x_input[1]
    # lambda2 = x_input[2]
    # lambda3 = x_input[3]
    mu1 = x_input[1]
    mu2 = x_input[2]
    mu3 = x_input[3]

    x = reshape(x_input[4:end], :, 1)

    o = one(T)
    z = zero(T)

    # --- vector field ---
    Fx = gradE(reshape(x, :, 3))
    Fx = reshape(Fx, 3N, 1)

    # --- Translation generators ---
    T1 = vcat(fill(o, N, 1), fill(z, 2N, 1))
    T2 = vcat(fill(z, N, 1), fill(o, N, 1), fill(z, N, 1))
    T3 = vcat(fill(z, 2N, 1), fill(o, N, 1))

    T1x = (hcat(fill(o, 1, N), fill(z, 1, 2N)) * x)
    T2x = (hcat(fill(z, 1, N), fill(o, 1, N), fill(z, 1, N)) * x)
    T3x = (hcat(fill(z, 1, 2N), fill(o, 1, N)) * x)

    # --- Rotation generators ---
    # I1xfix = vcat(-x_fix[N+1:2N, :], x_fix[1:N, :], fill(z, N, size(x_fix, 2)))
    # I2xfix = vcat(fill(z, N, size(x_fix, 2)), -x_fix[2N+1:end, :], x_fix[N+1:2N, :])
    # I3xfix = vcat(x_fix[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x_fix[1:N, :])

    # I1x = vcat(-x[N+1:2N, :], x[1:N, :], fill(z, N, size(x_fix, 2)))
    # I2x = vcat(fill(z, N, size(x_fix, 2)), -x[2N+1:end, :], x[N+1:2N, :])
    # I3x = vcat(x[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x[1:N, :])

    Fx_ext = Fx +
             mu1 * T1 + mu2 * T2 + mu3 * T3
    #  +lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    # _ = vcat(T1x, T2x, T3x, Fx_ext)

    dFx = hessE(reshape(x, :, 3))

    dFx = hcat(fill(z, 3N, 3), dFx)
    dFx = vcat(fill(z, 3, 3N + 3), dFx)


    IN = Diagonal(fill(o, N))

    ZNN = fill(z, N, N)

    Adjusts = [
        ZNN (interval(0.0))*IN (interval(0.0))*IN
        (interval(0.0))*IN ZNN (interval(0.0))*IN
        (interval(0.0))*IN (interval(0.0))*IN ZNN
    ]

    Adjusts = hcat(T1, T2, T3, Adjusts)

    Adjusts = vcat(
        hcat(fill(z, 1, 3), transpose(T1)),
        hcat(fill(z, 1, 3), transpose(T2)),
        hcat(fill(z, 1, 3), transpose(T3)), Adjusts)

    return dFx + Adjusts
end

### Initial configuration and parameters
σ = 1.0
ϵ = 1.0
α = 12.0
λ = 6.0
N = 256
L = 6σ
rc = 2.5σ

p = (σ, ϵ, α, λ, N, L, rc)
X, L, a = build_fcc_block(σ)


ff = ODEFunction(g_pbc!, jac=h_pbc!)
@show @time x_SS = solve(SteadyStateProblem(ff, X, p), DynamicSS(QNDF()), abstol=1e-6, reltol=1e-6)
x_ic = X
x_st = reshape(x_SS[:], :, 3)
println("residual norm after optimization: ", norm(g_pbc(x_st, p), Inf))

### Check that there are only three zero eigenvalues corresponding to translations
# eigen(h_pbc(X, p))

### The proof 
r0 = 1e-7
u = interval.(X, r0; format=:midpoint)

Kx, Ky, Kz = pbc_shift_matrices(X, N, 6.0)
check_pbc_shifts_cutoff(u, N, 6.0, 2.5, Kx, Ky, Kz)

σ = interval(1.0)
ϵ = interval(1.0)
α = interval(12.0)
λ = interval(6.0)
N = 256
L = interval(6.0)
rc = interval(2.5)
p = (σ, ϵ, α, λ, N, L, rc)


pI = (σ, ϵ, α, λ, N, L, rc, Kx, Ky, Kz)

duI = g_LJ_pbc_fixedshifts(u, pI)
JacI = h_LJ_pbc_fixedshifts(u, pI)

gradE = x -> -g_LJ_pbc_fixedshifts(x, pI)
hessE = x -> -h_LJ_pbc_fixedshifts(x, pI)

center(X) = X .- mean(X; dims=1)
u = center(u)
x = center(X)

F_int = x -> extended_Grad_pbc(x, interval(u), pI, gradE)
DF_int = x -> extended_Hess_pbc(x, interval(u), pI, gradE, hessE)

F_int([interval(zeros(3)); reshape(interval(x), :, 1)])
gradE(interval(x))

r = get_proof([interval(zeros(3)); reshape(interval(x), :, 1)], F_int, DF_int, 9.99938970585269e-7)
