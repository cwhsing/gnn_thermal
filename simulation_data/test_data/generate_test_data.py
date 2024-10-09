import h5py
import torch
from torch_geometric.data import Data
from fipy import CellVariable, Grid2D, TransientTerm, ExplicitDiffusionTerm

# Create the 2D grid for the 3x4 board
nx, ny = 30, 40
dx, dy = 0.1, 0.1
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# Define the patch locations (as tuples of (x, y) coordinates of the lower-left corners)
patch_locations = [(1.0, 1.0)]

# Initialize temperature field
T_initial = 300.0  # Initial temperature of the entire board
T_patch = 350.0    # Fixed temperature for the patch
T = CellVariable(mesh=mesh, value=T_initial)

# Set up the diffusion equation and use the explicit solver
alpha = 1.0  # Thermal diffusivity
eq = TransientTerm() == ExplicitDiffusionTerm(coeff=alpha)

# Time-stepping parameters
# CFL condition: timeStepDuration <= 1.0 * dx**2 / (4 * alpha)
timeStepDuration = 0.0025
steps = 800 + 1  # Simulate 2 seconds with 800 steps


# Function to apply a fixed temperature patch of size 1x1
def apply_patch(T, patch_x, patch_y, mesh, nx_patch=10, ny_patch=10):
    x, y = mesh.cellCenters
    patch_condition = ((x > patch_x) & (x < patch_x + nx_patch * dx) &
                       (y > patch_y) & (y < patch_y + ny_patch * dy))
    T.constrain(T_patch, where=patch_condition)


# Create an HDF5 file to save the data
with h5py.File('gnn_test_data.h5', 'w') as h5f:
    # Save edge_index once, as it is constant for all simulations
    # Create edges: Each node connects to its 4 neighbors (left, right, up, down)
    edges = []
    for i in range(ny):  # Iterate over rows
        for j in range(nx):  # Iterate over columns
            node_index = i * nx + j  # FiPy uses row-major order (ny -> nx)

            if j > 0:  # Connect to left neighbor
                left_neighbor = i * nx + (j - 1)
                edges.append([node_index, left_neighbor])
                edges.append([left_neighbor, node_index])  # Bidirectional

            if j < nx - 1:  # Connect to right neighbor
                right_neighbor = i * nx + (j + 1)
                edges.append([node_index, right_neighbor])
                edges.append([right_neighbor, node_index])  # Bidirectional

            if i < ny - 1:  # Connect to top neighbor
                top_neighbor = (i + 1) * nx + j
                edges.append([node_index, top_neighbor])
                edges.append([top_neighbor, node_index])  # Bidirectional

            if i > 0:  # Connect to bottom neighbor
                bottom_neighbor = (i - 1) * nx + j
                edges.append([node_index, bottom_neighbor])
                edges.append([bottom_neighbor, node_index])  # Bidirectional

    # Convert to tensor and transpose for PyG compatibility (shape [2, num_edges])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Save edge_index in HDF5
    h5f.create_dataset('edge_index', data=edge_index.numpy())

    # Loop through patch locations and save temperature data for each
    for patch_idx, (patch_x, patch_y) in enumerate(patch_locations):
        T.setValue(T_initial)  # Reset temperature field to initial value

        # Apply fixed temperature patch
        apply_patch(T, patch_x, patch_y, mesh)

        # Create a group for this patch in the HDF5 file
        patch_group = h5f.create_group(f'patch_{patch_idx}')
        temp_dset = patch_group.create_dataset('x', shape=(steps, nx * ny), dtype='float32')

        # Time-stepping loop for heat diffusion simulation
        for step in range(steps):
            if step > 0:
                eq.solve(var=T, dt=timeStepDuration)
            temp_dset[step, :] = T.value.copy()  # Store node features (temperature data)

