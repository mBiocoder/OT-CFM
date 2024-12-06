import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ot_cfm_optimizer(model, lr=1e-4):
    """Use Adam."""
    return torch.optim.Adam(model.parameters(), lr)

def train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, epochs=10000, plot_loss=False):
    """Train the CFM model."""
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch in dataloader:
            x1 = batch["x1"]
            x2 = batch["x2"]
            ot_cfm_optimizer.zero_grad()
            t, xt, ut = FM.sample_location_and_conditional_flow(x1, x2)
            vt = ot_cfm_model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            ot_cfm_optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
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
