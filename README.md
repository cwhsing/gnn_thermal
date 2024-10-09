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
where the time largely depends on the performance of the CPU. For a 2.4 GHz Intel Core i9, the typcial range is 5.0 ~ 5.5 seconds. The interactive mode allows you to inspect the details of the variables, such as the values of the temperature field.
![Image 1](https://raw.githubusercontent.com/lmcinnes/umap/master/images/umap_example_fashion_mnist1.png)

## Binary Data (class 5 & 7)
- Data: **dataset/fdata_57.npy**  
shape of fdata_57.npy: (14000, 28, 28)  

- Label: **dataset/flabel_57.npy**  
shape of flabel_57.npy: (14000)  
*notice: the labels are changed to **0 & 1** instead of the original 5 & 7*  

**Code example of loading the data into torch tensor:**
```
import numpy as np
import torch

x, y = np.load('fdata_57.npy'), np.load('flabel_57.npy')

x_train, x_test = torch.tensor(x[:12000])/255, torch.tensor(x[12000:])/255
y_train, y_test = torch.tensor(y[:12000])/255, torch.tensor(y[12000:])/255
```

**Both data and label are placed in the order of class 0 & 1.**  
```
x_train[:6000]: class 0, x_train[6000:]: class 1
y_train[:6000]: class 0, y_train[6000:]: class 1

x_test[:1000]: class 0, x_test[1000:]: class 1
y_test[:1000]: class 0, y_test[1000:]: class 1
```
   
## Ternary Data (class 5 & 7 & 9)
- Data: **dataset/fdata_579.npy**  
shape of fdata_579.npy: (21000, 28, 28)  

- Label: **dataset/flabel_579.npy**  
shape of flabel_579.npy: (21000)  
*notice: the labels are changed to **0 & 1 & 2** instead of the original 5 & 7 & 9*  

**Code example of loading the data into torch tensor:**
```
import numpy as np
import torch

x, y = np.load('fdata_579.npy'), np.load('flabel_579.npy')

x_train, x_test = torch.tensor(x[:18000])/255, torch.tensor(x[18000:])/255
y_train, y_test = torch.tensor(y[:18000])/255, torch.tensor(y[18000:])/255
```

**Both data and label are placed in the order of class 0 & 1 & 2**  
```
x_train[:6000]: class 0, x_train[6000:12000]: class 1, x_train[12000:18000]: class 2
y_train[:6000]: class 0, y_train[6000:12000]: class 1, y_train[12000:18000]: class 2

x_test[:1000]: class 0, x_test[1000:2000]: class 1, x_test[2000:3000]: class 2
y_test[:1000]: class 0, y_test[1000:2000]: class 1, y_test[2000:3000]: class 2
```

- You can customize another binary dataset comprising only class 7 & 9 or class 5 & 9 from this ternary one.  
- Judging from the UMAP above, I assume that class 5 & 7 will make the problem complex enough.

## Notice
***Since the data are placed in order, you have to shuffle them first before training.***

## Binary Classification (TTN-MPS, N=49, chi=2)
**parameter count = 686**

![Image 2](https://raw.githubusercontent.com/cwhsing/MPS-FashionMNIST/master/plot/fmnist57_chi2.png)

## Ternary Classification (TTN-MPS, N=49, chi=2)
**parameterr count = 686**

![Image 3](https://raw.githubusercontent.com/cwhsing/MPS-FashionMNIST/master/plot/fmnist579_chi2.png)
