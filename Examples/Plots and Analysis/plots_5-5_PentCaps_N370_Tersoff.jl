#### Plots for (5,5)-nanotube with pentagonal caps and N = 370 atoms using the Tersoff potential.
using GeometryOptimizationCAPs
using IntervalArithmetic
using CairoMakie
using JLD2

using Statistics

using UnPack, Printf, LaTeXStrings, CairoMakie

### Load the data saved from the validation script.
data = load("data/5-5_PentCaps_N370_Tersoff.jld2")
x_newton = data["x_newton"]
r = inf(data["r"])


connectivity, _ = get_5_5_connectivity_odd(33)
x = reshape(x_newton[7:end], :, 3)
x = center_nanotube_armchair(x)
x = interval.(x, r; format=:midpoint)


### Compute a reference radius using a central ring of atoms.
radii = sqrt.(x[:, 1] .^ 2 + x[:, 2] .^ 2)
reference_radius = mean(radii[181:190])

radii = sqrt.(x[:, 1] .^ 2 + x[:, 2] .^ 2)
reference_radius = mean(radii[181:190])
sup(radii[181:190][1])

common_decimal_places([sup(reference_radius), inf(reference_radius)])

### Functions to compute radii, bonds and angles.
function radii_plot_signed_deviation(
    x1, range, reference_radius, whisker;
    output_path="my_figure.pdf", scale_tol=1e-8
)

    # --- Custom log-linear transformation ---
    function smooth_symlog(y; t=scale_tol)
        sign.(y) .* ifelse.(abs.(y) .< t, abs.(y) / t, log10.(abs.(y) ./ t) .+ 1)
    end

    # --- Color palette ---
    colors = [
        "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#7570B3",
        "#E69F00", "#D55E00", "#E1E171", "#8B008B", "#A0522D",
    ]

    # --- Prepare ring data ---
    N = size(x1, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)

    upper_bound = zeros(Float64, number_of_rings)
    lower_bound = zeros(Float64, number_of_rings)
    mid_bound = zeros(Float64, number_of_rings)

    for i in 1:number_of_rings
        if i ∉ range
            continue
        end
        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]
        ring_intervals = get_radii(ring_coords) .- interval(reference_radius)
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        upper_bound[i] = maximum(sup.(ring_intervals))
        lower_bound[i] = minimum(inf.(ring_intervals))
        mid_bound[i] = minimum(mid.(ring_intervals))
    end

    # --- Set up figure and axis ---
    fig = Figure(size=(2000, 600), dpi=600, backgroundcolor=:white)
    half_rings = Int((number_of_rings - 1) / 2)

    # Define physical tick values and their transformed positions
    tick_physical = [-1e-4, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-4]
    tick_positions = smooth_symlog(tick_physical)

    tick_labels = [
        L"-10^{-4}",
        L"-10^{-6}",
        L"-10^{-8}",
        L"0",
        L"10^{-8}",
        L"10^{-6}",
        L"10^{-4}"
    ]



    ax = Axis(
        fig[1, 1],
        xlabel=L"\text{Index of cross section}",
        ylabel=L"\Delta r~\text{[\AA]}",
        titlesize=18,
        xlabelsize=55,
        ylabelsize=55,
        xticklabelsize=45,
        yticklabelsize=45,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        yticks=(tick_positions, tick_labels),
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

    # --- Plot each ring's deviations ---
    for i in 1:number_of_rings
        if i ∉ range
            continue
        end

        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]
        ring_intervals = get_radii(ring_coords) .- interval(reference_radius)
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        for j in 1:10
            x = i - half_rings - 1
            y_mid = smooth_symlog(mid.(ring_intervals[j]))
            y_lo = smooth_symlog(inf.(ring_intervals[j]))
            y_hi = smooth_symlog(sup.(ring_intervals[j]))

            scatter!(ax, [x], [y_mid], color=colors[1], markersize=10)
            rangebars!(
                ax,
                [x],
                [y_lo],
                [y_hi],
                color=colors[1],
                linewidth=5,
                whiskerwidth=whisker,
                alpha=0.9,
                transparency=true
            )
        end
    end

    # --- Add smoothed midline ---
    xvals = -half_rings:half_rings
    mid_transformed = smooth_symlog.(mid.(mid_bound))
    lines!(ax, xvals, mid_transformed, color="gray70", linewidth=2)

    save(output_path, fig)
    return fig
end

function compute_interval_archair_bonds(x_interval, connectivity, number_of_rings)

    horizontal_bonds_connect, rising_bonds_connect = sort_armchair_bonds(mid.(x_interval), connectivity)

    horizontal_bonds = zeros(eltype(x_interval), number_of_rings * 5)
    rising_bonds = zeros(eltype(x_interval), number_of_rings * 10)


    # ### Each 5 bonds corresponds to a ring
    for i in 1:number_of_rings*5

        indx = [horizontal_bonds_connect[i][1] horizontal_bonds_connect[i][2]]

        horizontal_bonds[i] = sqrt(sum((x_interval[indx[1], :] .- x_interval[indx[2], :]) .^ 2))

    end

    # ### Each 10 bonds corresponds to a ring
    for i in 1:number_of_rings*10

        indx = [rising_bonds_connect[i][1] rising_bonds_connect[i][2]]

        rising_bonds[i] = sqrt(sum((x_interval[indx[1], :] .- x_interval[indx[2], :]) .^ 2))

    end

    return transpose(reshape(horizontal_bonds, 5, :)), transpose(reshape(rising_bonds, 10, :))

end

function sort_armchair_angles(x, connectivity)
    ### Ruturns the indices of horizontal and rising bonds for each ring. Tube is assumed to be aligend to the z-axis.

    N = size(x, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)

    above_angle_connect = zeros(Int, N, 2)
    middle_angle_connect = zeros(Int, N, 2)
    below_angle_connect = zeros(Int, N, 2)

    for i in 1:number_of_rings
        for j in 1:10

            index = rings_indices[i, j]

            x_neighbours = [connectivity[index, :] x[connectivity[index, :], :]]
            ordered_neighbours = sortslices(x_neighbours, dims=1, by=x -> x[4])


            above_angle_connect[index, :] .= [Int.(ordered_neighbours[2, 1])
                Int.(ordered_neighbours[3, 1])]
            middle_angle_connect[index, :] .= [Int.(ordered_neighbours[3, 1]);
                Int.(ordered_neighbours[1, 1])]
            below_angle_connect[index, :] .= [Int.(ordered_neighbours[1, 1]);
                Int.(ordered_neighbours[2, 1])]

        end
    end

    return above_angle_connect, middle_angle_connect, below_angle_connect

end

function angle_between(p1, p2, p3)
    # Calculate the vectors
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the dot product and magnitudes
    dot_product = dot(v1, v2)
    mag_v1 = sqrt(dot(v1, v1))
    mag_v2 = sqrt(dot(v2, v2))

    # Calculate the angle in radians
    angle_rad = acos(dot_product / (mag_v1 * mag_v2))

    # # Convert to degrees
    # angle_deg = rad2deg(angle_rad)

    return angle_rad
end

function compute_interval_archair_angles(x_interval, connectivity)


    above_angle_connect, middle_angle_connect, below_angle_connect = sort_armchair_angles(mid.(x_interval), connectivity)


    N = size(x_interval, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)


    above_angles = zeros(eltype(x_interval), number_of_rings, 10)
    middle_angles = zeros(eltype(x_interval), number_of_rings, 10)
    below_angles = zeros(eltype(x_interval), number_of_rings, 10)


    # ### Each 10 bonds corresponds to a ring
    for i in 1:number_of_rings

        for j in 1:10

            index = rings_indices[i, j]
            particle = x_interval[index, :]

            ### Above angle     
            particle_middle = x_interval[above_angle_connect[index, 1], :]
            particle_above = x_interval[above_angle_connect[index, 2], :]
            above_angles[i, j] = angle_between(particle, particle_middle, particle_above)

            ### Middle angle
            particle_above = x_interval[middle_angle_connect[index, 1], :]
            particle_below = x_interval[middle_angle_connect[index, 2], :]
            middle_angles[i, j] = angle_between(particle, particle_above, particle_below)

            ### Below angle
            particle_below = x_interval[below_angle_connect[index, 1], :]
            particle_middle = x_interval[below_angle_connect[index, 2], :]
            below_angles[i, j] = angle_between(particle, particle_below, particle_middle)



        end

    end

    return above_angles, middle_angles, below_angles

end

function plot_armchair_bonds(x, connectivity, numRings, range, reference; output_path="my_figure.pdf")

    horizontal_bonds, rising_bonds = compute_interval_archair_bonds(x, connectivity, numRings)

    half_rings = Int((numRings - 1) / 2)

    ### Color pallete
    colors = [
        # "#000000",  # black
        # "#999999",  # grey
        "#0072B2",  # deep blue
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#CC79A7",  # muted purple
        "#7570B3",  # soft purple
        "#E69F00",  # gold/orange
        "#D55E00",  # orange-red
        "#E1E171",   # softened yellow (was F0E442)
        "#8B008B",  # dark magenta (new)
        "#A0522D",  # warm brown   (new)
    ]

    fig = Figure(
        size=(2000, 600),
        dpi=600,
        backgroundcolor=:white
    )
    ax = Axis(fig[1, 1];
        xlabel=L"\text{Index of cross section}",
        ylabel=L"\text{Bond length [\AA]}",
        # yscale=log10,
        titlesize=16,
        xlabelsize=55,
        ylabelsize=55,
        xticklabelsize=45,
        yticklabelsize=45,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        ytickformat=ys -> [@sprintf("%.3f", y) for y in ys],
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



    whisker = 0
    # marker = :utriangle

    for i in 2:numRings-1

        if i in range


            for j in 1:5

                scatter!(ax, [i - half_rings - 1], [mid.(horizontal_bonds[i, j])], color=colors[2], markersize=20)
                rangebars!(
                    ax,
                    [i - half_rings - 1],
                    inf.(horizontal_bonds[i, j]),
                    sup.(horizontal_bonds[i, j]),
                    color=colors[2],
                    linewidth=8,
                    whiskerwidth=whisker,
                    alpha=0.7,
                    transparency=true
                )

            end


        end

    end


    for i in 2:numRings

        if i in range


            if i != collect(range)[end]

                for j in 1:10

                    scatter!(ax, [i - half_rings - 1 + 0.5], [mid.(rising_bonds[i, j])], color=colors[2], markersize=20)
                    rangebars!(
                        ax,
                        [i - half_rings - 1],
                        inf.(rising_bonds[i, j]),
                        sup.(rising_bonds[i, j]),
                        color=colors[2],
                        linewidth=8,
                        whiskerwidth=whisker,
                        alpha=0.7,
                        transparency=true
                    )

                end


            end

        end

    end

    save(output_path, fig)
    fig

end

### Plot the radii.
radii_plot_signed_deviation(x, 1:31, reference_radius, 0.0; output_path="signed_difference_radius_5-5_PentCaps_N370_Tersoff.pdf", scale_tol=1e-9)

### Plot the bonds.
numRings = 31
horizontal_bonds, rising_bonds = sort_armchair_bonds(mid.(x), connectivity)

plot_armchair_bonds(x, connectivity, numRings, 3:29, 0.0; output_path="bonds_plot_5-5_PentCaps_N370_Tersoff.pdf")


### Plot the angles.
above_angles, middle_angles, below_angles = compute_interval_archair_angles(x, connectivity)


plot_armchair_angles(x, connectivity, numRings, 1:31, 0.0; output_path="angles_plot_5-5_PentCaps_N370_Tersoff.pdf")
