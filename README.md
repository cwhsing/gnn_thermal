# gnn_thermal
Simulator for thermal diffusion based on MeshGraphNets.

## Getting Started
Ensure to install the latest release of Anaconda so that all the basic reuquired dependencies will be automatically satisfied. Under a conda environment, install the latest release of PyTorch that matches your CUDA version. For generating data with Finite Element Method (FEM), we are using FiPy, which can be installed by:
```
pip install fipy
```
Manual of FiPy: https://www.ctcms.nist.gov/fipy/documentation/manual.html

For implementing Graph Neural Networks (GNNs) with PyTorch, we are using PyG, which can be installed by:
```
pip install torch-geometric
pip install torch-scatter
pip install torch-sparse
```
where the last two packages, while optional, are stronly reconmmended as they are commonly used together with PyG. For more details please refer to:
https://pytorch-geometric.readthedocs.io/en/latest/

## Simulation with FiPy (FEM)
To make later GNN implementation more manageable, we study a simpler heat diffusion case in which a 1(cm)x1(cm) patch with fixed temperature of 350K is placed on a 3(cm)x4(cm) board with initial temperature of 300K, thermal diffusivity of 1.0, and zero-flux Neumann boundary condition. The mesh is a 30x40 regular grid consisting of 1200 0.1(cm)x0.1(cm) cells:
```
# Create a 2D grid
nx, ny = 30, 40
dx, dy = 0.1, 0.1
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
```
The temperature field is initialized by:
```
T = CellVariable(mesh=mesh, value=300.0)  # Initial temperature of 300K
```
and the patch properties are set by:
```
# Define the coordinates of the internal patch (e.g., a small square region)
x, y = mesh.cellCenters
patch_condition = (x > 1.5) & (x < 2.5) & (y > 1.5) & (y < 2.5)

# Apply fixed temperature to the internal patch (e.g., 500K)
T.constrain(350.0, where=patch_condition)
```
We then set up the diffusion equation and adopt the explicit solver:
```
alpha = 1.0  # Thermal diffusivity
eq = TransientTerm() == ExplicitDiffusionTerm(coeff=alpha)
```
Since explicit solver is subject to the CFL condition, the largest stable time step is 0.0025 second. We found no significant difference when gradually decreasing the time step and therefore adopt the largest value. To make later GNN implementation more manageable, we simulate only for 800 steps, i.e. the first 2 seconds of the heat diffusion process. This can be done by:
```
# Time-stepping loop
# CFL condition: timeStepDuration <= 1.0 * dx**2 / (4 * alpha)
timeStepDuration = 0.0025
steps = 800 + 1
for step in range(steps):
    if step > 0:
        eq.solve(var=T, dt=timeStepDuration)
```
Implicit solver, on the other hand, is not subject to the CFL condition, but the accuracy with the same time step is worse than that of explicit solver. In principle, implicit solver is harder to code, but in FiPy this can easily be done by setting the equation as:
```
eq = TransientTerm() == DiffusionTerm(coeff=alpha)
```
In this project, however, we stick to explicit solver for generating high quality data.

### Example 1
Here we show an example in which the plot is shown and saved only after solving for all steps. Using the command:
```
python -i simulation_data/example_1/mesh2D_step-800.py
```
should show the final plot and give the following lines:
```
FEM simulation CPU time: 5.33 seconds
>>>
```
where the time largely depends on the performance of the CPU. For a 2.4 GHz Intel Core i9, the typcial range is 5.0 ~ 5.5 seconds. The interactive mode allows you to inspect the details of the variables, such as the values of the temperature field. The saved plot should look like

<img src="https://github.com/cwhsing/gnn_thermal/blob/main/simulation_data/example_1/step_800.png?raw=true" width=50% height=50%>

### Example 2
The second expample shows how to make an animation of the diffusion process by saving the plots for all time steps and combining them. Simply run:
```
python simulation_data/example_2/mesh2D_all-steps.py
```
and you will get 801 figures. To create an GIF animation with each figure as a frame, run:
```
python simulation_data/example_2/create_gif.py
```
Setting duration=20 gives you an FPS of 50, which is the maximally supported value in most viewers and browsers. The resulting GIF should look like

<img src="https://github.com/cwhsing/gnn_thermal/blob/main/simulation_data/example_2/data_19.gif?raw=true" width=50% height=50%>

## Data Generation
FiPy codes can readily be incorporated with PyG for generating time-series graphs. Note that in creating dataset files, HDF5 format should always be used as it can handle graph data much more effieciently. In particular, the size of the resulting dataset will be significantly smaller and loading the data to GPU will not incur memory overhead.

The training dataset includes 25 data, each with different patch location. To ensure maximal spatial diversity and avoid duplication due to symmetry, we manually crafted the patch locations as follows:
```
# Define the patch locations (as tuples of (x, y) coordinates of the lower-left corners)
patch_locations = [(0.1, 0.1), (0.1, 1.3), (0.1, 2.0), (0.3, 1.1), (0.3, 2.2),
                   (0.5, 0.9), (0.5, 2.4), (0.7, 0.4), (0.7, 1.7), (0.7, 2.9),
                   (0.9, 0.2), (0.9, 1.4), (0.9, 1.9), (1.1, 0.5), (1.1, 2.2),
                   (1.3, 0.7), (1.3, 2.0), (1.5, 1.2), (1.5, 1.5), (1.5, 2.7),
                   (1.7, 0.2), (1.7, 1.4), (1.7, 2.5), (1.9, 0.4), (1.9, 2.3)]
```
The initial and boundary conditions are always the same:
```
# Initialize temperature field
T_initial = 300.0  # Initial temperature of the entire board
T_patch = 350.0    # Fixed temperature for the patch
T = CellVariable(mesh=mesh, value=T_initial)
```
In our data graphs, the edges have no input feature and the indices are the same for all data. The nodes, on the other hand, are mesh centers with the corresponing temperature as the feature, which can be created by:
```
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
```
To generate the training dataset, simply run:
```
python simulation_data/training_data/generate_training_data.py
```
and you will get ```gnn_training_data.h5```
Similarly, you can get the test dataset ```gnn_test_data.h5``` with:
```
python simulation_data/test_data/generate_test_data.py
```
The test dataset contains one test data with the patch located at ```(1.0, 1.0)```, which is an unseen condition.

