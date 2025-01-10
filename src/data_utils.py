import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_adata(file_path, exclude_source="source_9"):
    """Loads and subsets the AnnData object."""
    adata = sc.read_h5ad(file_path)
    adata = adata[adata.obs["Metadata_Source"] != exclude_source].copy()  # bye-bye source_9
    return adata

class AnnDataSubsetDataset(Dataset):
    """Dataset class for subsets of AnnData objects."""
    def __init__(self, adata, source1, source2, use_pca=True):
        self.adata = adata
        self.source1 = source1
        self.source2 = source2
        self.use_pca = use_pca

        self.source1_data = self.adata[self.adata.obs["Metadata_Source"] == source1]
        self.source2_data = self.adata[self.adata.obs["Metadata_Source"] == source2]

        if use_pca:
            self.X_key = "X_pca"
            self.X1 = self.source1_data.obsm["X_pca"]
            self.X2 = self.source2_data.obsm["X_pca"]
        else:
            self.X_key = "X"
            self.X1 = self.source1_data.X
            self.X2 = self.source2_data.X

    def __len__(self):
        return min(len(self.source1_data), len(self.source2_data))

    def __getitem__(self, idx):
        idx1 = idx if len(self.source1_data) < len(self.source2_data) else np.random.randint(0, len(self.source1_data))
        idx2 =  idx if len(self.source1_data) > len(self.source2_data) else np.random.randint(0, len(self.source2_data))
        x1 = torch.tensor(self.X1[idx1], dtype=torch.float32)
        x2 = torch.tensor(self.X2[idx2], dtype=torch.float32)
        return {"x1": x1, "x2": x2}

class SingleSourceDataLoader(Dataset):
    """Dataset class for a single source."""
    def __init__(self, adata, source, use_pca=True):
        self.adata = adata
        self.source = source
        self.use_pca = use_pca

        self.source_data = self.adata[self.adata.obs["Metadata_Source"] == source]
        self.moas = self.source_data.obs.Metadata_moa
        self.cpds = self.source_data.obs.Metadata_JCP2022
        self.microscope = self.source_data.obs.Metadata_Microscope_Name

        if use_pca:
            self.X_key = "X_pca"
            self.X = self.source_data.obsm["X_pca"]

        else:
            self.X_key = "X"
            self.X = self.source_data.X

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        moa = self.moas[idx]
        cpd = self.cpds[idx]
        microscope = self.microscope[idx]

        # Replace NaN with a default value "NA"
        moa = moa if not pd.isna(moa) else "NA"
        cpd = cpd if not pd.isna(cpd) else "NA"
        microscope = microscope if not pd.isna(microscope) else "NA"

        return {"x": x, "moa": moa, "cpd": cpd, "microscope" : microscope}

def create_training_dataloader(adata, source1, source2, batch_size=64, use_pca=True):
    """Create the training data loader."""
    dataset = AnnDataSubsetDataset(adata, source1, source2, use_pca)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_dataloader(adata, source, batch_size=64, use_pca=True):
    """Create the dataloader"""
    dataset = SingleSourceDataLoader(adata, source, use_pca)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
