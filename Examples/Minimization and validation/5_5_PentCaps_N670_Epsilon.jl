using GeometryOptimizationCAPs
using IntervalArithmetic
using Statistics
using CairoMakie
using JLD2
using LinearAlgebra

using LineSearches
using ForwardDiff
using Optim

using UnPack, Printf, LaTeXStrings, CairoMakie

### A proof of the (5,5)-nanotube with pentagonal caps and N = 670 atoms using the epsilon potential. with epsilon equal to 1e-5. We minimize the energy and validate the simulation. 

### For this we use quadruple precision.
setprecision(BigFloat, 128)


connectivity, x_initial = get_5_5_connectivity_odd(63)

b = 1.44
θ = 2π / 3
kb = 1
kθ = 1

epsilon = 0.00001

e = x -> epsilon_energy(x, connectivity, b, θ, epsilon)

algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, x_initial, method=algo; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

F = x -> extended_Grad_epsilon(x, x_BFGS, connectivity, b, θ, epsilon)
DF = x -> extended_Hess_epsilon(x, x_BFGS, connectivity, b, θ, epsilon)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-10, maxiter=10)[1]

x_BFGS = center_nanotube_armchair(x_newton[7:end])
x_newton = [zeros(6); reshape(x_BFGS, :, 1)]

F = x -> extended_Grad_epsilon(x, x_BFGS, connectivity, b, θ, epsilon)
DF = x -> extended_Hess_epsilon(x, x_BFGS, connectivity, b, θ, epsilon)
x_newton = newton_method(x -> (F(x), DF(x)), BigFloat.(x_newton); tol=1.0e-30, maxiter=10)[1]

jldsave("5_5_PentCaps_N670_bond144_$epsilon.jld2"; x_newton, epsilon)

# F_int = x -> extended_Grad_epsilon(x, interval.(reshape(x_BFGS, :, 1)), connectivity, interval(b), interval(θ), interval(epsilon))
# DF_int = x -> extended_Hess_epsilon(x, interval.(reshape(x_BFGS, :, 1)), connectivity, interval(b), interval(θ), interval(epsilon))
# r = get_proof(x_newton, F_int, DF_int, 5.6068923423001035e-12)
### Note that this proof will take a long time to compute, as we are using quadruple precision and interval arithmetic.


# ### save the data and the radius
# # save("data/5_5_PentCaps_N670_epsilon.jld2", "x_newton", x_newton, "r", )
