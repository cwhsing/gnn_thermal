import time
from fipy import *
import matplotlib.pyplot as plt

# Start CPU time measurement
start_time = time.time()

# Create a 2D grid
nx, ny = 30, 40
dx, dy = 0.1, 0.1
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# Initialize the temperature field
T = CellVariable(mesh=mesh, value=300.0)  # Initial temperature of 300K

# Define the coordinates of the internal patch (e.g., a small square region)
x, y = mesh.cellCenters
patch_condition = (x > 1.5) & (x < 2.5) & (y > 1.5) & (y < 2.5)

# Apply fixed temperature to the internal patch (e.g., 500K)
T.constrain(350.0, where=patch_condition)

# Set up the diffusion equation and use the explicit solver
alpha = 1.0  # Thermal diffusivity
eq = TransientTerm() == ExplicitDiffusionTerm(coeff=alpha)

# Time-stepping loop
# CFL condition: timeStepDuration <= 1.0 * dx**2 / (4 * alpha)
timeStepDuration = 0.0025
steps = 800 + 1
for step in range(steps):
    if step > 0:
        eq.solve(var=T, dt=timeStepDuration)

# End CPU time measurement
run_time = time.time() - start_time
print(f"FEM simulation CPU time: {run_time:.2f} seconds")

# Plot the final solution
viewer = Viewer(vars=T, datamin=300.0, datamax=350.0)
viewer.plot()

# Adjust figure configurations
time = timeStepDuration * step
plt.title(f"step: {step:03d}    time: {time:.4f}s", fontsize=13, pad=10)
fig = plt.gcf()  # Get the current figure
cbar = fig.axes[-1]  # Access the colorbar axis
cbar.set_ylabel("temperature (K)", fontsize=13)

# Save the figure of the final step
plt.savefig(f"step_{step:03d}.png", dpi=220)
