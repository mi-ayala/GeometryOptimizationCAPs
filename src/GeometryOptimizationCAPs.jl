module GeometryOptimizationCAPs

using LinearAlgebra, ForwardDiff, Statistics, Distances, Random

using LineSearches, Optim, JLD2

using UnPack, Parameters, Printf, LaTeXStrings, CairoMakie

using IntervalArithmetic, RadiiPolynomial, StaticArrays

include("connectivities.jl")
include("harmonic_functions.jl")
include("tersoff_functions.jl")
include("epsilon_functions.jl")
include("proof_functions.jl")
include("radii_bonds_angles_functions.jl")
include("lattice_functions.jl")

@with_kw struct Tersoff_updated_parameters{R}
    a::R = 1393.6
    b::R = 430.0 ###
    λ₁::R = 3.4879
    λ₂::R = 2.2119
    β::R = 0.00000015724
    n::R = 0.72751
    h::R = -0.93 ###
    c::R = 38049.0
    d::R = 4.3484
end

@with_kw struct Tersoff_parameters{r}
    a::r = 1393.6
    b::r = 346.74
    λ₁::r = 3.4879
    λ₂::r = 2.2119
    β::r = 0.00000015724
    n::r = 0.72751
    h::r = -0.57058
    c::r = 38049.0
    d::r = 4.3484
end

### Exporting connectivities
export get_5_5_connectivity_even, get_connectivity_10_0_pentagonal,
    get_connectivity_10_0_hexagonal, get_connectivity_10_0_mixed, get_5_5_connectivity_odd

### Exporting Harmonic potential Functions
export harmonic_energy, Grad, extended_Grad, extended_Hess, Energy_Grad

### Exporting Tersoff potential Functions
export Tersoff_energy, Energy_Grad_Tersoff, extended_Hess_Tersoff, extended_Grad_Tersoff, Hess_Tersoff

### Exporting Epsilon potential Functions
export epsilon_energy, Grad_epsilon, extended_Grad_epsilon, extended_Hess_epsilon, Energy_Grad_epsilon

### Exporting Proof Functions
export newton_method, find_maximum_r_star, get_proof

### Exporting Radii, Bonds, Angles Functions
export center_nanotube_armchair, center_nanotube_zigzag, get_ring_indices, get_radii,
    radii_plot, sort_armchair_bonds, common_decimal_places

### Exporting Lattice Functions
export cubic_points, pts_to_mat, postprocess!, sc_positions, bcc_positions, fcc_positions,
    e_LJ, g_LJ!, g_LJ, h_LJ!, h_LJ, extended_Grad_LJ, extended_Hess_LJ,
    nn_rank_plot, coordination_plot

### Exporting Tersoff parameters
export Tersoff_parameters, Tersoff_updated_parameters

end # module GeometryOptimizationCAPs
