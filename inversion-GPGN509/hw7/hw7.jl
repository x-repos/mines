using CSV
using DataFrames
using CairoMakie
# Read the CSV file
df = CSV.File("Z_GR_NPHI_RHOB_CALI.txt",delim=',', ignorerepeated=true) |> DataFrame

# Convert columns
z = df[:, 1] .* 0.0003048  # Convert to kilometers
gr = df[:, 2]
nphi = df[:, 3]
rhob = df[:, 4]
cali = df[:, 5];

fig = Figure(size=(800, 600))
ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"f_X(x)", yreversed=true)
ax2 = Axis(fig[1, 2], xlabel=L"y", yreversed=true)
ax3 = Axis(fig[1, 3], xlabel=L"x", yreversed=true)
ax4 = Axis(fig[1, 4], xlabel=L"y", yreversed=true)
lines!(ax1, gr,   z, color=:black)
lines!(ax2, cali, z, color=:green)
lines!(ax3, nphi, z, color=:red)
lines!(ax4, rhob, z, color=:blue)
fig


σ_v = 0.2
