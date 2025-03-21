import torch
import scanpy as sc
import numpy as np
import pandas as pd
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torch.utils.data import DataLoader
import torch.nn as nn

class TorchWrapper(nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, c):
        super().__init__()
        self.model = model
        self.c = c  # One-hot encoding of source domain

    def forward(self, t, x, *args, **kwargs):
        time = t.view(-1, 1).expand(x.shape[0], -1)  # [batch_size, 1]
        source_one_hot = self.c.expand(x.shape[0], -1)  # [batch_size, num_sources]
        
        # **Only concatenate if the model does NOT already do it internally**
        if hasattr(self.model, "expects_raw_input") and self.model.expects_raw_input:
            x = torch.cat([x, source_one_hot, time], dim=-1)  
        
        return self.model(x, source_one_hot, time)
    
    
    
def transport_pooled(adata, ot_cfm_model, target_source="source_2", batch_size=64, device="cpu", save_adata_path=None):
    """Transport each source separately, conditioning on its one-hot vector."""
    
    num_sources = len(adata.obs["Metadata_Source"].unique()) - 1
    source_list = [src for src in adata.obs["Metadata_Source"].unique() if src != target_source]

    # Create DataLoaders for each source (excluding target)
    source_dataloaders = {
        source: create_dataloader(adata, source, batch_size=batch_size)
        for source in source_list
    }

    transported_data = []
    annot_combined = []

    for source in source_list:
        source_idx = source_list.index(source)  # Index for one-hot encoding
        one_hot_source = torch.eye(num_sources)[source_idx].to(device)  
        source_dataloader = source_dataloaders[source]

        # Wrap model for NeuralODE
        wrapped_model = TorchWrapper(ot_cfm_model, one_hot_source)
        node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint")

        for batch in source_dataloader:
            source_batch = batch["x"].to(device)  
            #source_cpd = batch["cpd"]  

            with torch.no_grad():
                traj = node.trajectory(
                    source_batch.float(),
                    t_span=torch.linspace(0, 1, 100),
                ).cpu()

            x_transported = traj[-1].detach().numpy()
            del traj  # Free memory

            transported_data.append(x_transported)

            # Create annotation for transported data
            annot_batch = pd.DataFrame({
                "Metadata_Source": [f"{source}_to_{target_source}"] * len(x_transported),
                #"Metadata_JCP2022": source_cpd[:len(x_transported)]  
            })
            annot_combined.append(annot_batch)

    # Combine transported data & annotations
    X_combined = np.concatenate(transported_data, axis=0)
    annot_combined = pd.concat(annot_combined, axis=0).reset_index(drop=True)

    print(f"Final transported data shape: {X_combined.shape}")
    print(f"Final annotation shape: {annot_combined.shape}")

    # Fix var_names
    original_var_names = adata.var_names[:X_combined.shape[1]]  # Ensure correct dimensions

    full_adata = sc.AnnData(X=X_combined, obs=annot_combined, var=pd.DataFrame(index=original_var_names))
    full_adata.obsm["X_pca"] = full_adata.X.copy()

    if save_adata_path:
        full_adata.write(save_adata_path)

    return full_adata