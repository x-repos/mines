import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Grid parameters (set to match your model)
nx, nz = 600, 350        # number of grid points (x,z)
dx, dz = 0.002, 0.002    # spacing in km (or m, depending on your input)
ox, oz = 0.0, 0.0        # origin

# -----------------------------
# Load binaries
vp  = np.fromfile("vp.bin", dtype=np.float32).reshape(nx, nz)
rho = np.fromfile("ro.bin", dtype=np.float32).reshape(nx, nz)

# -----------------------------
# Plot velocity
plt.figure(figsize=(8,6))
plt.imshow(vp.T, extent=[ox, ox+nx*dx, oz+nz*dz, oz],
           cmap="seismic", aspect="auto")
plt.colorbar(label="Vp (km/s)")
plt.title("Velocity Model")
plt.xlabel("x [km]")
plt.ylabel("z [km]")
plt.savefig("vp.png", dpi=150)
# plt.show()

# -----------------------------
# Plot density
plt.figure(figsize=(8,6))
plt.imshow(rho.T, extent=[ox, ox+nx*dx, oz+nz*dz, oz],
           cmap="viridis", aspect="auto")
plt.colorbar(label="Density (kg/m^3)")
plt.title("Density Model")
plt.xlabel("x [km]")
plt.ylabel("z [km]")
plt.savefig("ro.png", dpi=150)
# plt.show()
