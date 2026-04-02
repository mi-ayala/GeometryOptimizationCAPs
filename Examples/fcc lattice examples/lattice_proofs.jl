using GeometryOptimizationCAPs

using LinearAlgebra
using StaticArrays
using Statistics
using Random
using JLD2

using IntervalArithmetic
using RadiiPolynomial

using CairoMakie
using Printf

# ### Load data for perfect lattice
# file = load("Examples/fcc lattice examples/data_lattice_optimization_perfect.jld2")
# x_ic = file["x_ic"]
# x_st = file["x_st"]
# p = file["p"]

### Load data for lattices with defect
# file = load("Examples/fcc lattice examples/data_lattice_defect.jld2")
# x_defect = file["x_defect"]
# x_defect_B = file["x_defect_B"]
# p_d = file["p_d"]

# x_defect = reshape(x_defect[:], :, 3)
# x_defect_B = reshape(x_defect_B[:], :, 3)

### Saddle proof
file = load("Examples/fcc lattice examples/data_lattice_saddle.jld2")
x_saddle = file["x_saddle"]
p_d = file["p_d"]

p_int = (interval(1.0), interval(1.0), interval(12.0), interval(6.0), 863)

gradE = x -> -g_LJ(x, p_int)
hessE = x -> -h_LJ(x, p_int)

F_int = x -> extended_Grad_LJ(x, interval(x_saddle), p_int, gradE)
DF_int = x -> extended_Hess_LJ(x, interval(x_saddle), p_int, gradE, hessE)
r = get_proof([interval(zeros(6)); reshape(interval(x_saddle), :, 1)], F_int, DF_int, 9.99938970585269e-8)


# ### Coordination plot for saddle point
# x_saddle_interval = interval.(x_saddle, 9.9994e-08, format=:midpoint)
# fig_coord_saddle = coordination_plot(
#     x_saddle_interval,
#     771,
#     42;
#     whisker=20,
#     output_path="coordination_plot_saddle.pdf",
# )
