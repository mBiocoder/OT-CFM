import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ot_cfm_optimizer(model, lr=1e-4):
    """Use Adam."""
    return torch.optim.Adam(model.parameters(), lr)

def train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, device, epochs=10000, plot_loss=False):
    """
    Train the OT-CFM model with one-hot encoded source vectors.
    """
    ot_cfm_model.to(device) 
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for source_batch, target_batch, source_one_hot in dataloader:
            # Move data to the appropriate device
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            source_one_hot = source_one_hot.to(device)

            # Forward pass 
            outputs = ot_cfm_model(source_batch, source_one_hot)
            loss = FM.compute_loss(outputs, target_batch)

            # Backprop
            ot_cfm_optimizer.zero_grad()
            loss.backward()
            ot_cfm_optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

        # Average loss for the epoch
        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]}")

    # Plot training loss curve if param set
    if plot_loss:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.show()

    return ot_cfm_model

def save_ot_cfm_model(model, optimizer, filepath):
    """Save the model."""
    torch.save({"model": model, "optimizer": optimizer}, filepath)

def load_ot_cfm_model(model, optimizer, filepath):
    """Load the model."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"].state_dict())
    optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
