import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ot_cfm_optimizer(model, lr=1e-4):
    """Use Adam."""
    return torch.optim.Adam(model.parameters(), lr)

"""def train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, device, epochs=50, plot_loss=True):
    #Train the OT-CFM model with one-hot encoded source vectors.
    
    # Training loop 
    loss_fn = torch.nn.MSELoss()
    all_losses = []  
    
    for epoch in range(epochs):
        epoch_loss = 0.0  
        # Loop over batches from dataloader
        for batch in dataloader:
            # Unpack the batch dictionary
            source_batch = batch["source"]["x"].to(device)
            target_batch = batch["target"]["x"].to(device)
            source_one_hot = batch["source_label"].to(device)

            # Sample time uniformly
            time = torch.rand(source_batch.shape[0]).to(device)  # [1]
            time = time.unsqueeze(-1)  # Shape:[1, 1]

            # Expand time to match
            time = time.unsqueeze(1).expand(-1, 50, -1) 

            # Forward pass
            outputs = ot_cfm_model(source_batch, source_one_hot, time)
            loss = loss_fn(outputs, target_batch)

            # Backpropagation
            ot_cfm_optimizer.zero_grad()
            loss.backward()
            ot_cfm_optimizer.step()

            epoch_loss += loss.item() 

        # Storing average loss for this epoch...
        avg_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
    
    # Plot training loss curve if param set
    if plot_loss:
        #loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), all_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss Curve")
        plt.grid()
        plt.show()

    return ot_cfm_model
"""


class TrainingTorchWrapper(nn.Module):
    """Wraps model for training to ensure inputs match expected format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, source_batch, source_one_hot, time):
        """
        Forward pass during training.

        Args:
            source_batch (torch.Tensor): Shape [batch_size, feature_dim]
            source_one_hot (torch.Tensor): Shape [batch_size, num_sources]
            time (torch.Tensor): Shape [batch_size, 1]

        Returns:
            torch.Tensor: Model output
        """
        # Concatenate inputs along the last dimension
        x = torch.cat([source_batch, source_one_hot, time], dim=-1)  # Shape: [batch_size, feature_dim + num_sources + 1]

        # Forward pass through the model
        return self.model(x, source_one_hot, time)
    
    
def train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, device, epochs=50, plot_loss=True):
    """
    Train the OT-CFM model with one-hot encoded source vectors.
    """
    # Wrap the model
    wrapped_model = TrainingTorchWrapper(ot_cfm_model).to(device)

    # Training loop 
    loss_fn = torch.nn.MSELoss()
    all_losses = []  
    
    for epoch in range(epochs):
        epoch_loss = 0.0  
        
        for batch in dataloader:
            source_batch = batch["source"]["x"].to(device)  # [batch_size, feature_dim]
            target_batch = batch["target"]["x"].to(device)  # [batch_size, feature_dim]
            source_one_hot = batch["source_label"].to(device)  # [batch_size, num_sources]
            time = torch.rand(source_batch.shape[0], 1).to(device)  # [batch_size, 1]

            # Forward pass using wrapped model
            outputs = wrapped_model(source_batch, source_one_hot, time)
            loss = loss_fn(outputs, target_batch)

            # Backpropagation
            ot_cfm_optimizer.zero_grad()
            loss.backward()
            ot_cfm_optimizer.step()

            epoch_loss += loss.item() 

        avg_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
    
    # Plot training loss curve if param set
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), all_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss Curve")
        plt.grid()
        plt.show()

    return wrapped_model


    
def save_ot_cfm_model(model, optimizer, filepath):
    """Save the model."""
    torch.save({"model": model, "optimizer": optimizer}, filepath)

def load_ot_cfm_model(model, optimizer, filepath):
    """Load the model."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"].state_dict())
    optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
    return 
