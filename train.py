import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import get_loader
from model import DenseGrid, SingleLODmodel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
max_epochs = 100
learning_rate = 1.0e-3
input_dim = 3  # input dimension is 3 (XYZ coordinates)
hidden_dim = 64
output_dim = 1  # output dimension is 1 (occupancy label)

# Initialize model
grid_structure = DenseGrid(base_lod=4, num_lod=5, feature_size=8, interpolation_type="trilinear")
model = SingleLODmodel(res=128, feature_size=8, interpolation_type="trilinear")
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Load data
data_dir = './processed'
batch_size = 32
data_loader = get_loader(data_dir, device, batch_size)

# Training loop
for epoch in range(max_epochs):
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    # Iterate over batches
    for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch"):
        # Move data to device
        inputs, targets = batch
        
        # Forward pass
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.unsqueeze(1).float())  
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss for the epoch
    epoch_loss = total_loss / len(data_loader.dataset)
    
    # Print epoch details
    print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model_single_lod.pth')

print("Training complete")
