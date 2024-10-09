import h5py
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

gpu_id = 0
torch.cuda.set_device(gpu_id)
torch.cuda.empty_cache()
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

bs, lr, epochs = 5, 1e-6, 100


class EdgeEmbedding(nn.Module):
    """Edge embedding function: Latent edge attributes based on node temperatures."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(EdgeEmbedding, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, node_i, node_j):
        # Input node temperatures are divided by 1000 for numerical stability
        input_features = torch.cat([node_i / 1000.0, node_j / 1000.0], dim=-1)
        return self.edge_mlp(input_features)


class NodeEmbedding(nn.Module):
    """Node embedding function: Latent node attributes based on node temperature and aggregated edge embeddings."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(NodeEmbedding, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features, aggregated_edge_embeddings):
        input_features = torch.cat([node_features / 1000.0, aggregated_edge_embeddings], dim=-1)
        return self.node_mlp(input_features)


class HeatFlux(nn.Module):
    """Edge function: Predict heat flux based on latent node attributes."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(HeatFlux, self).__init__()
        self.flux_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)  # Output is the heat flux
        )

    def forward(self, latent_node_i, latent_node_j):
        input_features = torch.cat([latent_node_i, latent_node_j], dim=-1)
        return self.flux_mlp(input_features)


class MeshGraphNet(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MeshGraphNet, self).__init__(aggr='mean')  # Mean aggregation for message passing
        self.edge_embedding = EdgeEmbedding(node_dim, edge_dim, hidden_dim)
        self.node_embedding = NodeEmbedding(node_dim, edge_dim, hidden_dim)
        self.heat_flux = HeatFlux(node_dim, edge_dim, hidden_dim)

    def forward(self, x, edge_index):
        row, col = edge_index  # sender (row), receiver (col)

        # Step 1: Compute latent edge attributes (Edge Embedding)
        edge_attr = self.edge_embedding(x[row], x[col])

        # Step 2: Propagate messages (using latent edge attributes)
        node_latents = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Step 3: Compute heat flux (from latent node embeddings)
        flux = self.heat_flux(node_latents[row], node_latents[col])

        return flux

    def message(self, x_j, edge_attr):
        # Messages passed to the nodes are the latent edge attributes
        return edge_attr

    def update(self, aggr_out, x):
        # Update latent node attributes based on aggregated edge embeddings
        return self.node_embedding(x, aggr_out)


def update_temperature(x, flux, edge_index, dt=0.0025*1000):
    """Update the node temperatures based on the computed heat flux."""
    row, col = edge_index  # sender and receiver nodes
    # Update temperatures with the flux (simplified update rule)
    delta_temperature = torch.zeros_like(x)
    delta_temperature.index_add_(0, col, flux)  # Accumulate fluxes to receiver nodes
    x_updated = x + delta_temperature * dt  # Update temperature using time step (dt)
    return x_updated


def mae_loss(predicted_temperature, true_temperature):
    """Compute the MAE loss over the whole temperature field."""
    criterion = nn.L1Loss()
    return criterion(predicted_temperature, true_temperature)


def mse_loss(predicted_temperature, true_temperature):
    """Compute the MSE loss over the whole temperature field."""
    criterion = nn.MSELoss()
    return criterion(predicted_temperature, true_temperature)


class HDF5GraphDataset(Dataset):
    """Custom Dataset to load node features (temperature data) and edge_index from an HDF5 file."""
    def __init__(self, hdf5_file):
        # Open the HDF5 file
        self.hdf5_file = hdf5_file
        self.file = h5py.File(hdf5_file, 'r')

        # Load edge_index (same for all patches)
        self.edge_index = torch.tensor(self.file['edge_index'][:], dtype=torch.long)

        # Get the number of patches
        self.num_patches = len([key for key in self.file.keys() if key.startswith('patch_')])

    def __len__(self):
        # Return the number of patch locations
        return self.num_patches

    def __getitem__(self, idx):
        # Get the patch data group
        patch_group = self.file[f'patch_{idx}']

        # Load temperature data (node features) for all time steps
        temp_data = torch.tensor(patch_group['x'][:], dtype=torch.float32)

        # Return as PyTorch Geometric Data object
        return Data(x=temp_data.T, edge_index=self.edge_index)

    def __iter__(self):
        # Ensure the index within range in iteration
        for idx in range(self.__len__()):
            yield self.__getitem__(idx) 


def train_and_eval():
    # Initialize the dataset from the HDF5 file
    train_dataset = HDF5GraphDataset('gnn_training_data.h5')
    test_dataset = HDF5GraphDataset('gnn_test_data.h5')
    
    # Set up the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the model and optimizer
    model = MeshGraphNet(node_dim=1, edge_dim=1, hidden_dim=128).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, train_mae = 0, 0

        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()

            batch_loss, batch_mae = 0, 0  # Initialize loss and MAE for the current batch

            # Iterate over the 800 time steps (data.x has shape [1200*bs, 801])
            num_steps = data.x.shape[1]  # 801

            for step in range(num_steps - 1):
                current_step_data = data.x[:, step].unsqueeze(1)  # Shape [6000, 1]
                next_step_data = data.x[:, step + 1].unsqueeze(1)  # Ground truth for next step

                # Forward pass: compute heat flux
                flux = model(current_step_data, data.edge_index)

                # Update node temperatures using heat flux
                updated_temperature = update_temperature(current_step_data, flux, data.edge_index)

                # Compute loss with respect to the next time step
                step_loss = mse_loss(updated_temperature, next_step_data)
                step_loss.backward()  # Accumulate gradients for all steps

                # Compute MAE with respect to the next time step
                step_mae = mae_loss(updated_temperature, next_step_data)

                # Accumulate the loss and MAE for the current batch
                batch_loss += step_loss.item()
                batch_mae += step_mae.item()


            # Update model parameters after processing all time steps for the batch
            opt.step()

            # Add the average loss and MAE for this batch to the total loss
            train_loss += batch_loss / (num_steps - 1)  # Average loss over time steps
            train_mae += batch_mae / (num_steps - 1)

        # Evaluate the model on test data
        model.eval()
        test_mae = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                
                batch_mae = 0
                num_steps = data.x.shape[1]  # 801

                for step in range(num_steps - 1):
                    current_step_data = data.x[:, step].unsqueeze(1)  # Shape [1200, 1]
                    next_step_data = data.x[:, step + 1].unsqueeze(1)  # Ground truth for next step

                    # Forward pass: compute heat flux
                    flux = model(current_step_data, data.edge_index)

                    # Update node temperatures using heat flux
                    updated_temperature = update_temperature(current_step_data, flux, data.edge_index)

                    # Compute MAE with respect to the next time step
                    step_mae = mae_loss(updated_temperature, next_step_data)
                    batch_mae += step_mae.item()

                test_mae += batch_mae / (num_steps - 1)

        print(f'Epoch {epoch + 1:03d}/{epochs}, '
              f'MSE_Loss_train: {train_loss / len(train_loader):.6f}, '
              f'MAE_train: {train_mae / len(train_loader):.4f}, '
              f'MAE_test: {test_mae / len(test_loader):.4f}')

    # Save the model
    # torch.save(model.state_dict(), 'gns_model.pth')


def simulator():
    # Load the trained model
    model = MeshGraphNet(node_dim=1, edge_dim=1, hidden_dim=128)
    model.load_state_dict(torch.load('gns_model.pth', weights_only=True))
    model.to(device)

    # Initialize the dataset from the HDF5 file
    test_dataset = HDF5GraphDataset('gnn_test_data.h5')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize a figure for plotting
    plt.ion()  # Interactive mode on for live updates during simulation

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            num_steps = data.x.shape[1]  # 801
            for step in range(num_steps):
                current_step_data = data.x[:, step].unsqueeze(1)  # Shape [1200, 1]
                # Forward pass: compute heat flux
                flux = model(current_step_data, data.edge_index)
                # Update node temperatures using heat flux
                updated_temperature = update_temperature(current_step_data, flux, data.edge_index)
                # Update data.x with the predicted temperature for the next step
                data.x[:, step] = updated_temperature.squeeze()
                # Plot the current temperature distribution as a colormap
                plot_colormap(data, step)
            
            
# Set up the plotting function
def plot_colormap(data, step, nx=30, ny=40, dx=0.1, dy=0.1):
    temperature = data.x[:, step].cpu().numpy().reshape(ny, nx)  # Reshape node features to 2D grid
    plt.clf()  # Clear the previous plot
    plt.imshow(temperature, cmap='jet', origin='lower', extent=[0, nx*dx, 0, ny*dy], vmin=300, vmax=350)
    
    cbar = plt.colorbar()
    cbar.set_label('temperature (K)', fontsize=13)
    
    time = step * 0.0025
    plt.title(f"step: {step:03d}    time: {time:.4f}s", fontsize=13, pad=10)

    plt.xticks(np.linspace(0, nx * dx, 7))
    plt.yticks(np.linspace(0, ny * dy, 9))

    # plt.draw()
    # plt.pause(0.1)  # Pause to allow the plot to update

    plt.savefig(f"figure_{step:03d}.png", dpi=300)


if __name__ == '__main__':
    train_and_eval()
    # simulator()

