### Functions to generate initial lattice positions
cubic_points(n::Integer, d::Real) =
    vec([SVector(x, y, z) for x in 0:d:n*d, y in 0:d:n*d, z in 0:d:n*d])

function pts_to_mat(pts::AbstractVector{<:SVector{3}})
    N = length(pts)
    X = Matrix{Float64}(undef, 3, N)
    @inbounds for j in 1:N
        X[:, j] = pts[j]
    end
    return X
end

function postprocess!(X::AbstractMatrix{<:Real}; center::Bool=true, jitter::Real=0.0, rng=Random.GLOBAL_RNG)
    if center
        μ = mean(X; dims=2)
        X .-= μ
    end
    if jitter != 0.0
        X .+= jitter .* randn(rng, size(X))
    end
    return X
end

"""
    sc_positions(n, d; center=true, jitter=0.0, rng=Random.GLOBAL_RNG)

Simple Cubic positions as a 3×N matrix (columns are [x;y;z]).
"""
function sc_positions(n::Integer, d::Real; center::Bool=true, jitter::Real=0.0, rng=Random.GLOBAL_RNG)
    pts = cubic_points(n, d)
    X = pts_to_mat(pts)
    postprocess!(X; center, jitter, rng)
end

"""
    bcc_positions(n, d; center=true, jitter=0.0, rng=Random.GLOBAL_RNG)

Body-Centered Cubic positions as a 3×N matrix.
"""
function bcc_positions(n::Integer, d::Real; center::Bool=true, jitter::Real=0.0, rng=Random.GLOBAL_RNG)
    base = cubic_points(n, d)
    shift = SVector(0.5d, 0.5d, 0.5d)
    pts = vcat(base, [p .+ shift for p in base])
    pts = [p .* (2 / sqrt(3)) for p in pts]
    X = pts_to_mat(pts)
    postprocess!(X; center, jitter, rng)
end

"""
    fcc_positions(n, d; center=true, jitter=0.0, rng=Random.GLOBAL_RNG)

Face-Centered Cubic positions as a 3×N matrix.
"""
function fcc_positions(n::Integer, d::Real; center::Bool=true, jitter::Real=0.0, rng=Random.GLOBAL_RNG)
    base = cubic_points(n, d)
    s1 = SVector(0.0, 0.5d, 0.5d)
    s2 = SVector(0.5d, 0.0, 0.5d)
    s3 = SVector(0.5d, 0.5d, 0.0)
    pts = vcat(base,
        [p .+ s1 for p in base],
        [p .+ s2 for p in base],
        [p .+ s3 for p in base])
    pts = [p .* sqrt(2) for p in pts]
    X = pts_to_mat(pts)
    postprocess!(X; center, jitter, rng)
end

### LJ Functions. Energy, gradient and hessian
function e_LJ(u, p)
    σ, ϵ, α, λ, N = p

    u = transpose(reshape(u, :, 3))

    r = zeros(eltype(u[1]), N, N)
    ener = zero(eltype(u[1]))

    pairwise!(r, Euclidean(), u, dims=2)

    @inbounds for i in 1:(N-1)
        for j in (i+1):N
            ener += 4 * ϵ * ((σ / r[i, j])^α - (σ / r[i, j])^λ)
        end
    end

    return ener / N
end

@inline _T(::Type{T}, x) where {T} = oftype(zero(T), x)

### -grad(Energy) for the Lennard-Jones potential
function g_LJ!(du, u, p, t)
    σ, ϵ, α, λ, N = p
    N = Int(N)

    T = eltype(u)

    σT = _T(T, σ)
    ϵT = _T(T, ϵ)

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

    X = x .- x'
    Y = y .- y'
    Z = z .- z'

    r2 = @. X * X + Y * Y + Z * Z

    @inbounds for i in 1:N
        r2[i, i] = one(T)
    end

    r = sqrt.(r2)

    four = interval(4)
    invN = inv(interval(N))

    σ6 = σT^6
    c = four * ϵT * σ6

    F = @. c * (λi * inv(r^λp2) - (αi * σ6) * inv(r^αp2))

    @inbounds for i in 1:N
        F[i, i] = zero(T)
    end

    s = vec(sum(F; dims=2))

    dux .= s .* x .- F * x
    duy .= s .* y .- F * y
    duz .= s .* z .- F * z

    du .*= -invN
    return nothing
end

g_LJ(u, p) = (du = similar(u); fill!(du, 0); g_LJ!(du, u, p, 1); du)

### Hessian of -Energy for the Lennard-Jones potential
function h_LJ!(Jac, u, p, t)
    σ, ϵ, α, λ, N = p
    N = Int(N)
    T = eltype(u)

    αi = α
    λi = λ

    αp2 = αi + interval(2)
    λp2 = λi + interval(2)

    @views x = u[1:N]
    @views y = u[N+1:2N]
    @views z = u[2N+1:3N]

    X = x .- x'
    Y = y .- y'
    Z = z .- z'

    r2 = @. X * X + Y * Y + Z * Z
    @inbounds for i in 1:N
        r2[i, i] = one(T)
    end
    r = sqrt.(r2)

    σ6 = σ^interval(6)
    c = interval(4) * ϵ * σ6

    f = @. c * (λi * inv(r^λp2) - (αi * σ6) * inv(r^αp2))
    a = @. c * (
        -λi * (λi + interval(2)) * inv(r^(λp2 + interval(2))) +
        +αi * (αi + interval(2)) * σ6 * inv(r^(αp2 + interval(2)))
    )

    @inbounds for i in 1:N
        f[i, i] = zero(T)
        a[i, i] = zero(T)
    end

    Xa = X .* a
    Ya = Y .* a

    XX = X .* Xa .+ f
    YY = Y .* Ya .+ f
    ZZ = Z .* Z .* a .+ f

    YX = Y .* Xa
    ZX = Z .* Xa
    ZY = Z .* Ya

    fill!(Jac, zero(T))

    @views begin
        X_x = Jac[1:N, 1:N]
        X_y = Jac[1:N, N+1:2N]
        X_z = Jac[1:N, 2N+1:3N]

        Y_y = Jac[N+1:2N, N+1:2N]
        Y_z = Jac[N+1:2N, 2N+1:3N]

        Z_z = Jac[2N+1:3N, 2N+1:3N]
    end

    @inbounds for i in 1:N
        X_x[i, i] = sum(@view XX[i, :])
        Y_y[i, i] = sum(@view YY[i, :])
        Z_z[i, i] = sum(@view ZZ[i, :])

        X_y[i, i] = sum(@view YX[i, :])
        X_z[i, i] = sum(@view ZX[i, :])
        Y_z[i, i] = sum(@view ZY[i, :])

        X_x[:, i] .-= XX[:, i]
        Y_y[:, i] .-= YY[:, i]
        Z_z[:, i] .-= ZZ[:, i]

        X_y[:, i] .-= YX[:, i]
        X_z[:, i] .-= ZX[:, i]
        Y_z[:, i] .-= ZY[:, i]
    end

    @views begin
        Jac[N+1:2N, 1:N] .= X_y
        Jac[2N+1:3N, 1:N] .= X_z
        Jac[2N+1:3N, N+1:2N] .= Y_z
    end

    Jac .*= -inv(interval(N))
    return nothing
end

h_LJ(u, p) = begin
    N = size(u, 1)
    T = eltype(u)
    Jac = zeros(T, 3N, 3N)
    h_LJ!(Jac, u, p, one(T))
    Jac
end

### Extended functions for computer-assisted proofs
function extended_Grad_LJ(x_input, x_fix, p, gradE)
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, :, 1)
    T = eltype(x_input)

    N = Int(size(x_fix, 1) ÷ 3)

    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    Fx = gradE(reshape(x, :, 3))
    Fx = reshape(Fx, 3N, 1)

    o = one(T)
    z = zero(T)

    T1 = vcat(fill(o, N, 1), fill(z, 2N, 1))
    T2 = vcat(fill(z, N, 1), fill(o, N, 1), fill(z, N, 1))
    T3 = vcat(fill(z, 2N, 1), fill(o, N, 1))

    T1x = (hcat(fill(o, 1, N), fill(z, 1, 2N)) * x)
    T2x = (hcat(fill(z, 1, N), fill(o, 1, N), fill(z, 1, N)) * x)
    T3x = (hcat(fill(z, 1, 2N), fill(o, 1, N)) * x)

    I1xfix = vcat(-x_fix[N+1:2N, :], x_fix[1:N, :], fill(z, N, size(x_fix, 2)))
    I2xfix = vcat(fill(z, N, size(x_fix, 2)), -x_fix[2N+1:end, :], x_fix[N+1:2N, :])
    I3xfix = vcat(x_fix[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x_fix[1:N, :])

    I1x = vcat(-x[N+1:2N, :], x[1:N, :], fill(z, N, size(x_fix, 2)))
    I2x = vcat(fill(z, N, size(x_fix, 2)), -x[2N+1:end, :], x[N+1:2N, :])
    I3x = vcat(x[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x[1:N, :])

    Fx_ext = Fx +
             mu1 * T1 + mu2 * T2 + mu3 * T3 +
             lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    bal1 = (transpose(x) * I1xfix)
    bal2 = (transpose(x) * I2xfix)
    bal3 = (transpose(x) * I3xfix)

    return vcat(bal1, bal2, bal3, T1x, T2x, T3x, Fx_ext)
end

function extended_Hess_LJ(x_input, x_fix, p, gradE, hessE)
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, :, 1)
    T = eltype(x_input)

    N = Int(size(x_fix, 1) ÷ 3)

    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    o = one(T)
    z = zero(T)

    Fx = gradE(reshape(x, :, 3))
    Fx = reshape(Fx, 3N, 1)

    T1 = vcat(fill(o, N, 1), fill(z, 2N, 1))
    T2 = vcat(fill(z, N, 1), fill(o, N, 1), fill(z, N, 1))
    T3 = vcat(fill(z, 2N, 1), fill(o, N, 1))

    T1x = (hcat(fill(o, 1, N), fill(z, 1, 2N)) * x)
    T2x = (hcat(fill(z, 1, N), fill(o, 1, N), fill(z, 1, N)) * x)
    T3x = (hcat(fill(z, 1, 2N), fill(o, 1, N)) * x)

    I1xfix = vcat(-x_fix[N+1:2N, :], x_fix[1:N, :], fill(z, N, size(x_fix, 2)))
    I2xfix = vcat(fill(z, N, size(x_fix, 2)), -x_fix[2N+1:end, :], x_fix[N+1:2N, :])
    I3xfix = vcat(x_fix[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x_fix[1:N, :])

    I1x = vcat(-x[N+1:2N, :], x[1:N, :], fill(z, N, size(x_fix, 2)))
    I2x = vcat(fill(z, N, size(x_fix, 2)), -x[2N+1:end, :], x[N+1:2N, :])
    I3x = vcat(x[2N+1:end, :], fill(z, N, size(x_fix, 2)), -x[1:N, :])

    Fx_ext = Fx +
             mu1 * T1 + mu2 * T2 + mu3 * T3 +
             lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    _ = vcat(transpose(x) * I1xfix, transpose(x) * I2xfix, transpose(x) * I3xfix, T1x, T2x, T3x, Fx_ext)

    dFx = hessE(reshape(x, :, 3))

    dFx = hcat(fill(z, 3N, 6), dFx)
    dFx = vcat(fill(z, 6, 3N + 6), dFx)

    IN = Diagonal(fill(o, N))
    ZNN = fill(z, N, N)

    Adjusts = [
        ZNN (-lambda1) * IN (lambda3) * IN
        (lambda1) * IN ZNN (-lambda2) * IN
        (-lambda3) * IN (lambda2) * IN ZNN
    ]

    Adjusts = hcat(I1x, I2x, I3x, T1, T2, T3, Adjusts)

    Adjusts = vcat(
        hcat(fill(z, 1, 6), transpose(I1xfix)),
        hcat(fill(z, 1, 6), transpose(I2xfix)),
        hcat(fill(z, 1, 6), transpose(I3xfix)),
        hcat(fill(z, 1, 6), transpose(T1)),
        hcat(fill(z, 1, 6), transpose(T2)),
        hcat(fill(z, 1, 6), transpose(T3)),
        Adjusts
    )

    return dFx + Adjusts
end

### Coordination plot
function nn_rank_plot(x::AbstractMatrix;
    k::Int=26,
    cube::NTuple{6,Float64}=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    max_lines::Int=50,
    savepath::Union{Nothing,String}=nothing)

    @assert size(x, 2) == 3 "x must be N×3 (columns = x,y,z)"
    N = size(x, 1)
    xlo, xhi, ylo, yhi, zlo, zhi = cube

    inside = [i for i in 1:N if (xlo <= x[i, 1] <= xhi) &&
              (ylo <= x[i, 2] <= yhi) &&
              (zlo <= x[i, 3] <= zhi)]
    isempty(inside) && error("No particles found inside the cube $(cube).")

    keff = min(k, N - 1)
    M = length(inside)
    D = Matrix{Float64}(undef, keff, M)

    dist = pairwise(Euclidean(), x, dims=1)
    for (j, i) in enumerate(inside)
        row = sort(view(dist, i, :))
        D[:, j] = collect(row[2:(keff + 1)])
    end

    fig = Figure(size=(1200, 800), backgroundcolor=:white)
    ax = Axis(
        fig[1, 1],
        xlabel="neighbor ranking",
        ylabel="distance",
        title="Nearest-neighbor distances for particles in unit cube",
    )

    r = 1:keff
    nlines = min(M, max_lines)
    for j in 1:nlines
        lines!(ax, r, D[:, j], color=(:gray, 0.35), linewidth=1)
        scatter!(ax, r, D[:, j], color=(:gray, 0.35), markersize=6)
    end

    med = vec(mapslices(median, D; dims=2))
    lines!(ax, r, med, color=:blue, linewidth=3)

    if savepath !== nothing
        path = splitext(savepath)[2] == "" ? string(savepath, ".png") : savepath
        save(path, fig)
        @info "Plot saved to $path"
    end

    display(fig)
    return D, inside
end

"""
    coordination_plot(X, center_idx, nneigh;
                      whisker=20,
                      output_path="coordination_plot.pdf")

Given an array of interval-valued coordinates `X` (either 3×N or N×3),
plots the distances (as intervals) from particle `center_idx` to its
`nneigh` closest neighbors.

- x-axis: neighbor index 1,2,…,nneigh (sorted by distance)
- y-axis: distance to the chosen particle (in Å, or your units)
.
"""
function coordination_plot(
    X::AbstractMatrix,
    center_idx::Integer,
    nneigh::Integer;
    whisker::Real=20,
    output_path::AbstractString="coordination_plot.pdf"
)
    coords =
        if size(X, 1) == 3 && size(X, 2) >= 1
            permutedims(X)
        elseif size(X, 2) == 3
            X
        else
            error("X must be 3×N or N×3 (interval-valued coordinates).")
        end

    N = size(coords, 1)
    @assert 1 <= center_idx <= N "center_idx out of bounds"
    @assert nneigh >= 1 "nneigh must be >= 1"
    nneigh = min(nneigh, N - 1)

    colors = [
        "#0072B2",
        "#56B4E9",
        "#009E73",
        "#CC79A7",
        "#7570B3",
        "#E69F00",
        "#D55E00",
        "#E1E171",
        "#8B008B",
        "#A0522D",
    ]

    center = coords[center_idx, :]

    dists = [
        sqrt((coords[j, 1] - center[1])^2 +
             (coords[j, 2] - center[2])^2 +
             (coords[j, 3] - center[3])^2)
        for j in 1:N
    ]

    neighbor_ids = collect(1:N)
    deleteat!(neighbor_ids, findfirst(==(center_idx), neighbor_ids))

    sorted_neighbors = sort(neighbor_ids; by=j -> mid(dists[j]))
    selected = sorted_neighbors[1:nneigh]
    selected_dists = dists[selected]

    fig = Figure(
        size=(2000, 700),
        dpi=600,
        backgroundcolor=:white,
    )

    ax = Axis(
        fig[1, 1],
        xlabel=L"\text{Neighbor index}",
        ylabel=L"\text{Distance from ref. atom}",
        titlesize=18,
        xlabelsize=50,
        ylabelsize=50,
        xticklabelsize=45,
        yticklabelsize=45,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        xtickalign=1,
        ytickalign=1,
        yticksmirrored=true,
        xticksmirrored=true,
        xticksize=15,
        yticksize=15,
        spinewidth=4,
        xtickwidth=4,
        ytickwidth=4,
    )

    for (k, d) in enumerate(selected_dists)
        m = mid(d)
        lo = inf(d)
        hi = sup(d)

        scatter!(ax, [k], [m],
            color=colors[1],
            markersize=30)

        rangebars!(
            ax,
            [k],
            [lo],
            [hi],
            color=colors[1],
            linewidth=1,
            whiskerwidth=whisker,
            alpha=0.7,
            transparency=true,
        )
    end

    save(output_path, fig)
    return fig
end
