#### Energy for (5,5)-nanotube with pentagonal caps and N = 360 atoms using the harmonic potential.
using GeometryOptimizationCAPs
using IntervalArithmetic
using CairoMakie
using Statistics
using JLD2
using UnPack, Printf, LaTeXStrings, CairoMakie

### Load the data saved from the validation script.
data = load("data/5-5_PentCaps_N360_Harmonic.jld2")
x_newton = data["x_newton"]
r = inf(data["r"])


x = reshape(x_newton[7:end], :, 3)
x = interval.(x, r; format=:midpoint)

connectivity, _ = get_5_5_connectivity_even(30)

### Compute total harmonic energy per atom.
N = size(x, 1)
b = 1.44
θ = 2π / 3
kb = 469
kθ = 63
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)
E = e(x) / N
common_decimal_places([sup.(E); inf.(E)])
sup(E)


# radii = sqrt.(x[:, 1] .^ 2 + x[:, 2] .^ 2)
# reference_radius = mean(radii[171:190])
# sup(radii[181:190][1])
# common_decimal_places([sup.(radii[171:190]); inf.(radii[171:190])])




