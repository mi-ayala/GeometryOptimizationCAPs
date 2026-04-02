using GeometryOptimizationCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic
using RadiiPolynomial

using JLD2

### A proof of the (5,5)-nanotube with pentagonal caps and N = 370 atoms using the harmonic potential. We minimize the energy and validate the simulation. All the other examples follow the same implementation structure.

### We start by minimizing the harmonic energy. Before using carbon parameters,it is easier to converge with all parameters set to 1.
b = 1
θ = 2π / 3
kb = 1
kθ = 1

### Construct the initial configuration for a (5,5) nanotube with an odd number of middle sections(rings).
connectivity, x_initial = get_5_5_connectivity_odd(33)
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)

### BFGS optimization stage.
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, x_initial, method=algo, g_tol=1e-6; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

### Newton refinement with the same parameters.
F = x -> extended_Grad(x, x_BFGS, connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_BFGS, connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)];
    tol=1.0e-13, maxiter=10)[1]

### Re-run Newton refinement with carbon parameters.
b = 1.44
θ = 2π / 3
kb = 469
kθ = 63

F = x -> extended_Grad(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_newton[7:end], :, 1)];
    tol=1.0e-13, maxiter=10)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]

### Validation step using interval arithmetic.
F_int = x -> extended_Grad(x, interval.(x_newton[7:end]), connectivity,
    interval(b), interval(θ), interval(kb), interval(kθ))
DF_int = x -> extended_Hess(x, interval.(x_newton[7:end]), connectivity,
    interval(b), interval(θ), interval(kb), interval(kθ))

### Use the following function to find the maximum validation radius.
# r = find_maximum_r_star(x_newton, F_int, DF_int, 9.999931334517948e-7, 1e-6)

r = get_proof(x_newton, F_int, DF_int, 9.999931334517948e-7)

### Save the data.
save("data/5-5_PentCaps_N370_Harmonic.jld2", "x_newton", x_newton, "r", r)

### The data is saved to be loaded and analyzed in the file `plots_5-5_PentCaps_N370_Harmonic.jl`, where we generate plots and examine the results.
