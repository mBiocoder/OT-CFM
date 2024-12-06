import numpy as np
import scanpy as sc
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
import torch
import pandas as pd
    
def transport(dataloader, ot_cfm_model, x1_label="source_3", device = "cpu", save_adata_path = None):
    """Transports data using trained OT-CFM."""
    ot_cfm_model.to(device)
    node = NeuralODE(torch_wrapper(ot_cfm_model), solver="dopri5", sensitivity="adjoint")

    #annot = []
    X_combined = []
    annot_combined = pd.DataFrame()

    # Transport source1 to source2
    for batch in dataloader:
      x1 = batch["x"]
      moa = batch["moa"]
      cpd = batch["cpd"]
      microscope = batch["microscope"]


      with torch.no_grad():
          traj = node.trajectory(
              x1.float().to(device),
              t_span=torch.linspace(0, 1, 100),
          ).cpu()
      x1_transported = traj[-1].detach().numpy()
      del traj

      # Concatenate data for source1 along with transported source1
      X_combined.append(np.concatenate([x1_transported, x1], axis=0))

      #create annotations for annData
      annot1 = {"Metadata_Source": f"{x1_label}_translated", "Metadata_moa" : moa, "Metadata_JCP2022" : cpd, "Metadata_Microscope_Name" : microscope}
      annot2 = {"Metadata_Source": x1_label , "Metadata_moa" : moa, "Metadata_JCP2022" : cpd, "Metadata_Microscope_Name" : microscope}
      annot_batch = pd.concat([pd.DataFrame(annot1), pd.DataFrame(annot2)], axis = 0)
      annot_combined = pd.concat([annot_combined, annot_batch], axis = 0)

    # Make AnnData object
    X_combined = np.concatenate(X_combined, axis=0)
    full_adata = sc.AnnData(X=X_combined, obs=annot_combined)
    full_adata.obsm["X_pca"] = full_adata.X.copy()

    if save_adata_path:
      #print("Writing translated adata to file...")
      full_adata.write(save_adata_path)

    return full_adata
