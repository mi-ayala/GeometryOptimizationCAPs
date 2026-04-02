#### Plots for (10,0)-nanotube with pentagonal caps and N = 660 atoms using the harmonic potential. 
using GeometryOptimizationCAPs
using IntervalArithmetic
using CairoMakie
using JLD2

using UnPack, Printf, LaTeXStrings, CairoMakie

### Load the data saved from the validation script.
data = load("data/10-0_PentCaps_N660_Harmonic.jld2")
x_newton = data["x_newton"]
r = inf(data["r"])

### Reshape the configuration and intervalize.
x = reshape(x_newton[7:end], :, 3)
x = interval.(x, r; format=:midpoint)

numRings = 31
connectivity, _ = get_connectivity_10_0_pentagonal(numRings)

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

### Plot of the radii for different sections of the nanotube
function zigzag_ring_indices(rings_indices)
    num_rings = size(rings_indices, 1)
    zigzag_ring_indices = [zeros(Int32, num_rings, 1) rings_indices]
    sequence = vec([i + j for i in 3:4:62, j in 0:1])

    for i in 1:num_rings

        if i ∈ sequence

            zigzag_ring_indices[i, 1] = 1

        else

            ring2 = rings_indices[i, vec([1, 3, 5, 7, 9])]
            ring3 = rings_indices[i, vec([2, 4, 6, 8, 10])]


            zigzag_ring_indices[i, 2:6] .= ring2
            zigzag_ring_indices[i, 7:end] .= ring3

        end


    end

    return zigzag_ring_indices
end

function radii_plot_zigzag(
    x1, range, n
    ; output_path="my_figure.pdf"
)

    ### Color pallete
    colors = [
        "#000000",  # black
        "#999999",  # grey
        "#0072B2",  # deep blue
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#CC79A7",  # muted purple
        "#7570B3",  # soft purple
        "#E69F00",  # gold/orange
        "#D55E00",  # orange-red
        "#E1E171",   # softened yellow 
        "#8B008B",  # dark magenta 
        "#A0522D",  # warm brown   
    ]

    ### Determine ring indices
    N = size(x1, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    rings_indices = zigzag_ring_indices(rings_indices)[:, 2:end]

    number_of_rings = size(rings_indices, 1)
    println("Number of rings: ", number_of_rings)
    half_rings = Int((number_of_rings) / 2)

    upper_bound = zeros(Float64, number_of_rings)
    lower_bound = zeros(Float64, number_of_rings)
    mid_bound = zeros(Float64, number_of_rings)

    ### Add padding in y axis
    for i in 1:number_of_rings

        if i ∉ range
            continue
        end

        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]

        ring_intervals = abs.(get_radii(ring_coords))
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        upper_bound[i] = maximum(mid.(ring_intervals))
        lower_bound[i] = minimum(mid.(ring_intervals))
        mid_bound[i] = maximum(mid.(ring_intervals))


    end

    fig = Figure(
        size=(1200, 600),
        dpi=600,
        backgroundcolor=:white
    )

    ax = Axis(
        fig[1, 1],
        xlabel=L"\text{Index of cross section}",
        ylabel=L"\text{Distance to tube's axis [\AA]}",
        titlesize=18,
        # yscale=log10,
        xlabelsize=40,
        ylabelsize=40,
        xticklabelsize=35,
        yticklabelsize=35,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        ytickformat=ys -> [@sprintf("%.*f", n, y) for y in ys],
        xtickalign=1,
        ytickalign=1,
        yticksmirrored=true,
        xticksmirrored=true,
        xticksize=15,
        yticksize=15,
        spinewidth=4,
        xtickwidth=4,
        ytickwidth=4
    )


    lines!(
        ax,
        range .- half_rings,
        upper_bound[range],
        color="gray70",
        linewidth=3,
    )

    lines!(
        ax,
        range .- half_rings,
        lower_bound[range],
        color="gray70",
        linewidth=3,
    )


    whisker = 20
    ###  Loop over each ring
    for i in 1:number_of_rings

        if i ∉ range
            continue
        end
        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]

        ring_intervals = abs.(get_radii(ring_coords))
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        for j in 1:10
            scatter!(ax, [i - half_rings], [mid.(ring_intervals[j])], color=colors[3], markersize=10)
            rangebars!(
                ax,
                [i - half_rings],
                [inf.(ring_intervals[j])],
                [sup.(ring_intervals[j])],
                color=colors[3],
                linewidth=2,
                whiskerwidth=whisker,
                alpha=0.9,
                transparency=true
            )
        end

    end

    save(output_path, fig)
    fig
end

radii_plot_zigzag(x, 11:16, 6; output_path="radius_10-0_PentCaps_N660_Harmonic_11_16.pdf")

radii_plot_zigzag(x, 23:28, 10; output_path="radius_10-0_PentCaps_N660_Harmonic_23_28.pdf")

radii_plot_zigzag(x, 33:37, 10; output_path="radius_10-0_PentCaps_N660_Harmonic_33_37.pdf")

radii_plot_zigzag(x, 45:49, 10; output_path="radius_10-0_PentCaps_N660_Harmonic_45_49.pdf")
