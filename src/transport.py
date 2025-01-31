import numpy as np
import scanpy as sc
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
import torch
import torch.nn as nn
import pandas as pd


#see notebook for updated code for this part...

"""class TorchWrapper(nn.Module):
    #Wraps model to torchdyn compatible format
    def __init__(self, model, c):
        super().__init__()
        self.model = model
        self.c = c

    def forward(self, t, x, *args, **kwargs):
        time = t.repeat(x.shape[0])[:, None]  # Repeat t for batch size
        return self.model(x, time, self.c)


def transport_pooled(adata, dataloader, ot_cfm_model, target_source="source_2", device="cpu", save_adata_path=None):

    num_sources = len(adata.obs["Metadata_Source"].unique())
    target_idx = list(adata.obs["Metadata_Source"].unique()).index(target_source)
    one_hot_target = torch.eye(num_sources)[target_idx].to(device)

    # Excluding the target source from the dataset (source 2)
    pooled_data = adata[adata.obs["Metadata_Source"] != target_source]

    # Wrap the model for NeuralODE
    wrapped_model = TorchWrapper(ot_cfm_model, one_hot_target)
    node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint")

    X_combined, annot_combined = [], pd.DataFrame()

    # Transport all sources (pooled) to the target source
    for batch in dataloader:
        x = batch["x"]
        moa = batch["moa"]
        cpd = batch["cpd"]
        microscope = batch["microscope"]

        with torch.no_grad():
            traj = node.trajectory(
                x.float(),
                t_span=torch.linspace(0, 1, 100),
            ).cpu()
        x_transported = traj[-1].detach().numpy()
        del traj

        # Concatenate transported and original data
        X_combined.append(np.concatenate([x_transported, x.cpu().numpy()], axis=0))

        # Create annotations for AnnData
        annot1 = {
            "Metadata_Source": [f"{src}_to_{target_source}" for src in source_labels],
            "Metadata_moa": moa,
            "Metadata_JCP2022": cpd,
            "Metadata_Microscope_Name": microscope,
        }
        annot2 = {
            "Metadata_Source": source_labels,
            "Metadata_moa": moa,
            "Metadata_JCP2022": cpd,
            "Metadata_Microscope_Name": microscope,
        }
        annot_batch = pd.concat([pd.DataFrame(annot1), pd.DataFrame(annot2)], axis=0)
        annot_combined = pd.concat([annot_combined, annot_batch], axis=0)

    # Combine transported data into an AnnData object
    X_combined = np.concatenate(X_combined, axis=0)
    full_adata = sc.AnnData(X=X_combined, obs=annot_combined)
    full_adata.obsm["X_pca"] = full_adata.X.copy()

    if save_adata_path:
        full_adata.write(save_adata_path)

    return full_adata

"""