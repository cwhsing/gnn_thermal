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

## The Architechture of the GNN Model 

## Training the GNN Model
Make sure you have the ```gnn_training_data.h5``` created and copied to the same directory having ```gns.py```. With the ```train_and_eval()``` function on, run:
```
python gns.py
```
or if on a server, run:
```
nohup python -u gns.py > log
```
The result of the first 100 epochs is:
```
Epoch 001/100, MSE_Loss_train: 0.454998, MAE_train: 0.6385, MAE_test: 0.4846
Epoch 002/100, MSE_Loss_train: 0.272070, MAE_train: 0.4739, MAE_test: 0.3270
Epoch 003/100, MSE_Loss_train: 0.147620, MAE_train: 0.3190, MAE_test: 0.1863
Epoch 004/100, MSE_Loss_train: 0.078994, MAE_train: 0.1848, MAE_test: 0.0737
Epoch 005/100, MSE_Loss_train: 0.050006, MAE_train: 0.0827, MAE_test: 0.0438
Epoch 006/100, MSE_Loss_train: 0.043883, MAE_train: 0.0394, MAE_test: 0.0970
Epoch 007/100, MSE_Loss_train: 0.047225, MAE_train: 0.0778, MAE_test: 0.1268
Epoch 008/100, MSE_Loss_train: 0.049543, MAE_train: 0.0958, MAE_test: 0.1300
Epoch 009/100, MSE_Loss_train: 0.048909, MAE_train: 0.0914, MAE_test: 0.1155
Epoch 010/100, MSE_Loss_train: 0.046538, MAE_train: 0.0736, MAE_test: 0.0931
Epoch 011/100, MSE_Loss_train: 0.044590, MAE_train: 0.0509, MAE_test: 0.0702
Epoch 012/100, MSE_Loss_train: 0.043653, MAE_train: 0.0320, MAE_test: 0.0540
Epoch 013/100, MSE_Loss_train: 0.043614, MAE_train: 0.0316, MAE_test: 0.0468
Epoch 014/100, MSE_Loss_train: 0.043794, MAE_train: 0.0359, MAE_test: 0.0450
Epoch 015/100, MSE_Loss_train: 0.043826, MAE_train: 0.0363, MAE_test: 0.0468
Epoch 016/100, MSE_Loss_train: 0.043703, MAE_train: 0.0339, MAE_test: 0.0497
Epoch 017/100, MSE_Loss_train: 0.043626, MAE_train: 0.0312, MAE_test: 0.0538
Epoch 018/100, MSE_Loss_train: 0.043588, MAE_train: 0.0286, MAE_test: 0.0572
Epoch 019/100, MSE_Loss_train: 0.043584, MAE_train: 0.0274, MAE_test: 0.0583
Epoch 020/100, MSE_Loss_train: 0.043600, MAE_train: 0.0283, MAE_test: 0.0592
Epoch 021/100, MSE_Loss_train: 0.043597, MAE_train: 0.0283, MAE_test: 0.0579
Epoch 022/100, MSE_Loss_train: 0.043590, MAE_train: 0.0276, MAE_test: 0.0561
Epoch 023/100, MSE_Loss_train: 0.043583, MAE_train: 0.0281, MAE_test: 0.0553
Epoch 024/100, MSE_Loss_train: 0.043585, MAE_train: 0.0285, MAE_test: 0.0543
Epoch 025/100, MSE_Loss_train: 0.043583, MAE_train: 0.0285, MAE_test: 0.0552
Epoch 026/100, MSE_Loss_train: 0.043586, MAE_train: 0.0280, MAE_test: 0.0560
Epoch 027/100, MSE_Loss_train: 0.043580, MAE_train: 0.0280, MAE_test: 0.0549
Epoch 028/100, MSE_Loss_train: 0.043577, MAE_train: 0.0283, MAE_test: 0.0543
Epoch 029/100, MSE_Loss_train: 0.043572, MAE_train: 0.0283, MAE_test: 0.0545
Epoch 030/100, MSE_Loss_train: 0.043573, MAE_train: 0.0279, MAE_test: 0.0554
Epoch 031/100, MSE_Loss_train: 0.043573, MAE_train: 0.0275, MAE_test: 0.0559
Epoch 032/100, MSE_Loss_train: 0.043578, MAE_train: 0.0277, MAE_test: 0.0547
Epoch 033/100, MSE_Loss_train: 0.043567, MAE_train: 0.0278, MAE_test: 0.0548
Epoch 034/100, MSE_Loss_train: 0.043568, MAE_train: 0.0276, MAE_test: 0.0551
Epoch 035/100, MSE_Loss_train: 0.043569, MAE_train: 0.0276, MAE_test: 0.0545
Epoch 036/100, MSE_Loss_train: 0.043567, MAE_train: 0.0277, MAE_test: 0.0542
Epoch 037/100, MSE_Loss_train: 0.043567, MAE_train: 0.0276, MAE_test: 0.0549
Epoch 038/100, MSE_Loss_train: 0.043565, MAE_train: 0.0276, MAE_test: 0.0539
Epoch 039/100, MSE_Loss_train: 0.043559, MAE_train: 0.0277, MAE_test: 0.0541
Epoch 040/100, MSE_Loss_train: 0.043561, MAE_train: 0.0277, MAE_test: 0.0537
Epoch 041/100, MSE_Loss_train: 0.043562, MAE_train: 0.0275, MAE_test: 0.0547
Epoch 042/100, MSE_Loss_train: 0.043562, MAE_train: 0.0270, MAE_test: 0.0551
Epoch 043/100, MSE_Loss_train: 0.043554, MAE_train: 0.0270, MAE_test: 0.0543
Epoch 044/100, MSE_Loss_train: 0.043553, MAE_train: 0.0272, MAE_test: 0.0536
Epoch 045/100, MSE_Loss_train: 0.043555, MAE_train: 0.0275, MAE_test: 0.0531
Epoch 046/100, MSE_Loss_train: 0.043552, MAE_train: 0.0276, MAE_test: 0.0532
Epoch 047/100, MSE_Loss_train: 0.043562, MAE_train: 0.0274, MAE_test: 0.0542
Epoch 048/100, MSE_Loss_train: 0.043550, MAE_train: 0.0269, MAE_test: 0.0542
Epoch 049/100, MSE_Loss_train: 0.043555, MAE_train: 0.0269, MAE_test: 0.0539
Epoch 050/100, MSE_Loss_train: 0.043547, MAE_train: 0.0272, MAE_test: 0.0528
Epoch 051/100, MSE_Loss_train: 0.043546, MAE_train: 0.0274, MAE_test: 0.0526
Epoch 052/100, MSE_Loss_train: 0.043549, MAE_train: 0.0273, MAE_test: 0.0533
Epoch 053/100, MSE_Loss_train: 0.043551, MAE_train: 0.0273, MAE_test: 0.0525
Epoch 054/100, MSE_Loss_train: 0.043543, MAE_train: 0.0273, MAE_test: 0.0529
Epoch 055/100, MSE_Loss_train: 0.043547, MAE_train: 0.0271, MAE_test: 0.0528
Epoch 056/100, MSE_Loss_train: 0.043542, MAE_train: 0.0268, MAE_test: 0.0539
Epoch 057/100, MSE_Loss_train: 0.043543, MAE_train: 0.0268, MAE_test: 0.0545
Epoch 058/100, MSE_Loss_train: 0.043540, MAE_train: 0.0267, MAE_test: 0.0534
Epoch 059/100, MSE_Loss_train: 0.043537, MAE_train: 0.0265, MAE_test: 0.0530
Epoch 060/100, MSE_Loss_train: 0.043538, MAE_train: 0.0269, MAE_test: 0.0520
Epoch 061/100, MSE_Loss_train: 0.043540, MAE_train: 0.0269, MAE_test: 0.0523
Epoch 062/100, MSE_Loss_train: 0.043539, MAE_train: 0.0270, MAE_test: 0.0522
Epoch 063/100, MSE_Loss_train: 0.043543, MAE_train: 0.0272, MAE_test: 0.0515
Epoch 064/100, MSE_Loss_train: 0.043532, MAE_train: 0.0269, MAE_test: 0.0528
Epoch 065/100, MSE_Loss_train: 0.043538, MAE_train: 0.0267, MAE_test: 0.0536
Epoch 066/100, MSE_Loss_train: 0.043541, MAE_train: 0.0266, MAE_test: 0.0522
Epoch 067/100, MSE_Loss_train: 0.043531, MAE_train: 0.0265, MAE_test: 0.0523
Epoch 068/100, MSE_Loss_train: 0.043529, MAE_train: 0.0264, MAE_test: 0.0526
Epoch 069/100, MSE_Loss_train: 0.043531, MAE_train: 0.0263, MAE_test: 0.0522
Epoch 070/100, MSE_Loss_train: 0.043527, MAE_train: 0.0263, MAE_test: 0.0524
Epoch 071/100, MSE_Loss_train: 0.043542, MAE_train: 0.0267, MAE_test: 0.0510
Epoch 072/100, MSE_Loss_train: 0.043525, MAE_train: 0.0265, MAE_test: 0.0518
Epoch 073/100, MSE_Loss_train: 0.043523, MAE_train: 0.0262, MAE_test: 0.0524
Epoch 074/100, MSE_Loss_train: 0.043526, MAE_train: 0.0267, MAE_test: 0.0530
Epoch 075/100, MSE_Loss_train: 0.043526, MAE_train: 0.0268, MAE_test: 0.0523
Epoch 076/100, MSE_Loss_train: 0.043530, MAE_train: 0.0264, MAE_test: 0.0508
Epoch 077/100, MSE_Loss_train: 0.043528, MAE_train: 0.0268, MAE_test: 0.0505
Epoch 078/100, MSE_Loss_train: 0.043523, MAE_train: 0.0267, MAE_test: 0.0513
Epoch 079/100, MSE_Loss_train: 0.043516, MAE_train: 0.0261, MAE_test: 0.0524
Epoch 080/100, MSE_Loss_train: 0.043521, MAE_train: 0.0270, MAE_test: 0.0529
Epoch 081/100, MSE_Loss_train: 0.043531, MAE_train: 0.0266, MAE_test: 0.0510
Epoch 082/100, MSE_Loss_train: 0.043532, MAE_train: 0.0266, MAE_test: 0.0526
Epoch 083/100, MSE_Loss_train: 0.043520, MAE_train: 0.0271, MAE_test: 0.0525
Epoch 084/100, MSE_Loss_train: 0.043515, MAE_train: 0.0266, MAE_test: 0.0516
Epoch 085/100, MSE_Loss_train: 0.043517, MAE_train: 0.0263, MAE_test: 0.0497
Epoch 086/100, MSE_Loss_train: 0.043521, MAE_train: 0.0265, MAE_test: 0.0505
Epoch 087/100, MSE_Loss_train: 0.043515, MAE_train: 0.0262, MAE_test: 0.0503
Epoch 088/100, MSE_Loss_train: 0.043517, MAE_train: 0.0261, MAE_test: 0.0513
Epoch 089/100, MSE_Loss_train: 0.043512, MAE_train: 0.0259, MAE_test: 0.0511
Epoch 090/100, MSE_Loss_train: 0.043515, MAE_train: 0.0263, MAE_test: 0.0517
Epoch 091/100, MSE_Loss_train: 0.043518, MAE_train: 0.0264, MAE_test: 0.0509
Epoch 092/100, MSE_Loss_train: 0.043533, MAE_train: 0.0266, MAE_test: 0.0526
Epoch 093/100, MSE_Loss_train: 0.043545, MAE_train: 0.0267, MAE_test: 0.0491
Epoch 094/100, MSE_Loss_train: 0.043516, MAE_train: 0.0263, MAE_test: 0.0506
Epoch 095/100, MSE_Loss_train: 0.043524, MAE_train: 0.0268, MAE_test: 0.0525
Epoch 096/100, MSE_Loss_train: 0.043514, MAE_train: 0.0264, MAE_test: 0.0504
Epoch 097/100, MSE_Loss_train: 0.043509, MAE_train: 0.0259, MAE_test: 0.0500
Epoch 098/100, MSE_Loss_train: 0.043507, MAE_train: 0.0260, MAE_test: 0.0499
Epoch 099/100, MSE_Loss_train: 0.043509, MAE_train: 0.0262, MAE_test: 0.0515
Epoch 100/100, MSE_Loss_train: 0.043507, MAE_train: 0.0262, MAE_test: 0.0507
```
where we can see that the model converges within ~20 epochs and the best model (corresponding to the lowest MAE_test) is saved with:
```
torch.save(model.state_dict(), 'gns_model.pth')
```
On a Nvidia RTX 4090 GPU, each epoch takes ~12 seconds.

We have several key findings:
- The fact that MAE_train is smaller than 0.03K and MAE_train is around 0.05K inidicates that decent accuracies can be achieved with the current model.
- The fact that MAE_test is only slightly above MAE_train indicates that the current model generalizes well.
- The fact that MAE_test does not increase over many training epochs indicates that there is no overfitting. 

## Simulate with the Saved Model and Test Data
Make sure you have the ```gns_model.pth``` saved and copied to the same directory having ```gns.py```. With the ```simulator()``` function on, run:
```
python gns.py
```
You will get 801 plots which can be combined into a GIF animation. The result should look like the file ```data_test.gif``` as shown below:

<img src="https://github.com/cwhsing/gnn_thermal/blob/main/data_test.gif?raw=true" width=50% height=50%>
