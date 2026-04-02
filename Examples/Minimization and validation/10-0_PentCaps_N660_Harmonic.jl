using GeometryOptimizationCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic

using JLD2

### A proof of the (10,0)-nanotube with pentagonal caps and N = 664 atoms using the harmonic potential. We minimize the energy and validate the simulation. All the other examples follow the same implementation structure.

b = 1.44
θ = 2π / 3
kb = 469
kθ = 63

numRings = 31
connectivity, x_initial = get_connectivity_10_0_pentagonal(numRings)
N = size(x_initial, 1)
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)

### BFGS optimization stage.
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, x_initial, method=algo, g_tol=1e-12; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

### Newton refinement with the same parameters.
F = x -> extended_Grad(x, x_BFGS, connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_BFGS, connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-13, maxiter=10)[1]

F = x -> extended_Grad(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_newton[7:end], :, 1)]; tol=1.0e-13, maxiter=10)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]

### Validation step using interval arithmetic.
F_int = x -> extended_Grad(x, interval.(x_newton[7:end]), connectivity, interval(b), interval(θ), interval(kb), interval(kθ))
DF_int = x -> extended_Hess(x, interval.(x_newton[7:end]), connectivity, interval(b), interval(θ), interval(kb), interval(kθ))

r = get_proof(x_newton, F_int, DF_int, 9.999389780282882e-8)

# ### Save the data 
# save("data/10-0_PentCaps_N660_Harmonic.jld2", "x_newton", x_newton, "r", r)

### The data is saved to be loaded and analyzed in the file `plots_5-5_PentCaps_N660_Harmonic.jl`, where we generate plots and examine the results.
