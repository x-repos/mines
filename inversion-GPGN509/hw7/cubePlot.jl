function plot3D(x, y, z, g, slice;
    direction="y",
    lx="x-axis", ly="y-axis", lz="z-axis",
    aspect_ratio=(1, 1, 1),
    azimuth=0.4, title="3D Plot", colorbar=false)

    fig = Figure()
    ax = Axis3(fig[1, 1]; 
        aspect=aspect_ratio,
        xlabel=lx, ylabel=ly, zlabel=lz,
        perspectiveness=0.5,
        azimuth=azimuth,
        title=title
    )

    cr = extrema(g)  # color range to use for all heatmaps

    # Determine the transformation based on the specified direction
    for i in slice
        if direction == "x"
            hm = heatmap!(ax, y, z, g[i, :, :]; 
                colorrange=cr, transparency=true, 
                colormap=:dense, transformation=(:yz, x[i]))
        elseif direction == "y"
            hm = heatmap!(ax, x, z, g[:, i, :]; 
                colorrange=cr, transparency=true, 
                colormap=:dense, transformation=(:xz, y[i]))
        elseif direction == "z"
            hm = heatmap!(ax, x, y, g[:, :, i]; 
                colorrange=cr, transparency=true, 
                colormap=:dense, transformation=(:xy, z[i]))
        else
            error("Invalid direction: choose from 'x', 'y', or 'z'")
        end

        # Add colorbar once for the first heatmap if colorbar=true
        if colorbar && i == slice[1]
            Colorbar(fig[1, 2], hm, width=10, height=Relative(0.6))
        end
    end

    # Set the axis limits
    xlims!(ax, minimum(x), maximum(x))
    ylims!(ax, minimum(y), maximum(y))
    zlims!(ax, minimum(z), maximum(z))

    return fig
end
