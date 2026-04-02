using GeometryOptimizationCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic

using JLD2

### A proof of the (5,5)-nanotube with pentagonal caps and N = 370 atoms using the Tersoff potential. We minimize the energy and validate the simulation. All the other examples follow the same implementation structure.


### We start by minimizing the harmonic energy. Before using Tersoff potential.
b = 1.4
θ = 2π / 3
kb = 1
kθ = 1

p = Tersoff_parameters()

connectivity, x_initial = get_5_5_connectivity_odd(33)
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
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-12, maxiter=10)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]


### Validation step using interval arithmetic.
F_int = x -> extended_Grad_Tersoff(x, interval.(x_newton[7:end]), p, connectivity)
DF_int = x -> extended_Hess_Tersoff(x, interval.(x_newton[7:end]), p, connectivity)

r = get_proof(x_newton, F_int, DF_int, 9.9212e-8)

# ### Save the data 
# save("data/5-5_PentCaps_N370_Tersoff.jld2", "x_newton", x_newton, "r", r)


### The data is saved to be loaded and analyzed in the file `plots_5-5_PentCaps_N370_Tersoff.jl`, where we generate plots and examine the results.


### The following part check that given the originla parameters for the cutoff of the Tersoff potential, every otom iteract with only its first neighbors.


x_tube = reshape(x_newton[7:end], :, 3)
x₀r_interval = interval.(x_tube, 9.9212e-8; format=:midpoint) ### This is radius from the proof

function smooth_cutoff3(r::Interval, R::Real, D::Real)
    lo = R - D
    hi = R + D

    r_lo, r_hi = inf(r), sup(r)

    if r_hi < lo
        return :one
    elseif r_lo > hi
        return :zero
    elseif (r_lo >= lo) && (r_hi <= hi)
        return :middle
    else
        return :mixed
    end
end

function pairwise_distances_and_cutoff(X::AbstractMatrix, R, D)
    @assert size(X, 2) == 3 "X must be N×3"
    N = size(X, 1)

    Tcoord = eltype(X)
    Tdiff = typeof(zero(Tcoord) - zero(Tcoord))
    Tdist = typeof(sqrt(zero(Tdiff)))

    if Tcoord <: Interval

        Dmat = Matrix{Tdist}(undef, N, N)
        Cmat = Matrix{Symbol}(undef, N, N)

        @inbounds for i in 1:N-1
            xi1, xi2, xi3 = X[i, 1], X[i, 2], X[i, 3]
            for j in i+1:N
                dx = xi1 - X[j, 1]
                dy = xi2 - X[j, 2]
                dz = xi3 - X[j, 3]

                r = sqrt(dx * dx + dy * dy + dz * dz)

                Dmat[i, j] = r
                Dmat[j, i] = r

                c = smooth_cutoff3(r, R, D)   # Symbol
                Cmat[i, j] = c
                Cmat[j, i] = c
            end
        end

        for i in 1:N
            Dmat[i, i] = zero(Tdist)
            Cmat[i, i] = :self
        end

        return Dmat, Cmat
    else

        Dmat = Matrix{Tdist}(undef, N, N)

        cutoff_example = smooth_cutoff3(zero(Tdist), R, D)
        Tcut = typeof(cutoff_example)
        Cmat = Matrix{Tcut}(undef, N, N)

        @inbounds for i in 1:N-1
            xi1, xi2, xi3 = X[i, 1], X[i, 2], X[i, 3]
            for j in i+1:N
                dx = xi1 - X[j, 1]
                dy = xi2 - X[j, 2]
                dz = xi3 - X[j, 3]

                r = sqrt(dx * dx + dy * dy + dz * dz)

                Dmat[i, j] = r
                Dmat[j, i] = r

                c = smooth_cutoff3(r, R, D)
                Cmat[i, j] = c
                Cmat[j, i] = c
            end
        end

        for i in 1:N
            Dmat[i, i] = zero(Tdist)
            Cmat[i, i] = zero(Tcut)
        end

        return Dmat, Cmat
    end
end

function check_three_neighbors(C::AbstractMatrix; expected::Int=3, tol::Real=1e-3)
    T = eltype(C)

    if T == Symbol
        # Tag-based logic (interval case)
        N = size(C, 1)
        counts_one = zeros(Int, N)
        bad_idx = Int[]

        @inbounds for i in 1:N
            cnt = 0
            for j in 1:N
                if i == j
                    continue  # skip diagonal (:self)
                end
                if C[i, j] === :one
                    cnt += 1
                end
            end
            counts_one[i] = cnt
            if cnt != expected
                push!(bad_idx, i)
            end
        end

        return bad_idx, counts_one
    else

        row_sums = vec(sum(C, dims=2))
        Tsum = eltype(row_sums)
        expected_T = convert(Tsum, expected)
        tol_T = convert(Tsum, tol)

        bad_idx = findall(i -> abs(row_sums[i] - expected_T) > tol_T,
            eachindex(row_sums))

        return bad_idx, row_sums
    end
end

### Candidate cutoff validation with parameters:
R = 1.95
D = 0.15

### An empty bad_indx means that all atoms in the candidate have exactly three interacting neighbors.
Dmat, Cmat = pairwise_distances_and_cutoff(x₀r_interval, R, D)
bad_idx, row_sums = check_three_neighbors(Cmat; expected=3)

