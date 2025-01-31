import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ot_cfm_optimizer(model, lr=1e-4):
    """Use Adam."""
    return torch.optim.Adam(model.parameters(), lr)

def train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, device, epochs=50, plot_loss=False):
    """
    Train the OT-CFM model with one-hot encoded source vectors.
    """
    
    # Training loop 
    loss_fn = torch.nn.MSELoss()
    all_losses = []  
    
    for epoch in range(epochs):
        epoch_loss = 0.0  
        # Loop over batches from the dataloader
        for batch in dataloader:
            # Unpack the batch dictionary
            source_batch = batch["source"]["x"].to(device)
            target_batch = batch["target"]["x"].to(device)
            source_one_hot = batch["source_label"].to(device)

            # Sample time uniformly
            time = torch.rand(source_batch.shape[0]).to(device)  # Shape: [64]
            time = time.unsqueeze(-1)  # Shape: [64, 1]

            # Expand time to match
            time = time.unsqueeze(1).expand(-1, 64, -1) 

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

    
def save_ot_cfm_model(model, optimizer, filepath):
    """Save the model."""
    torch.save({"model": model, "optimizer": optimizer}, filepath)

def load_ot_cfm_model(model, optimizer, filepath):
    """Load the model."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"].state_dict())
    optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
