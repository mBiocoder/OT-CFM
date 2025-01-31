import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
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

def create_dataloader(adata, source, batch_size=64, use_pca=True):
    """Create the dataloader"""
    dataset = SingleSourceDataLoader(adata, source, use_pca)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



class PooledDataset(torch.utils.data.Dataset):
    """
    Dataset for pooling all sources except a specified target source
    """
    def __init__(self, adata, target_source, batch_size, use_pca=True):
        self.adata = adata
        self.target_source = target_source
        self.batch_size = batch_size
        self.use_pca = use_pca

        # Separate target source and other sources
        self.target_data = adata[adata.obs["Metadata_Source"] == target_source]
        self.other_sources = adata[adata.obs["Metadata_Source"] != target_source]

        # Prepare PCA or raw features for target
        self.target_X = (
            self.target_data.obsm["X_pca"] if use_pca else self.target_data.X
        )
        self.target_moas = self.target_data.obs.Metadata_moa
        self.target_cpds = self.target_data.obs.Metadata_JCP2022
        self.target_microscope = self.target_data.obs.Metadata_Microscope_Name

        # Prepare data for each other source
        self.sources = {
            source: adata[adata.obs["Metadata_Source"] == source]
            for source in self.other_sources.obs["Metadata_Source"].unique()
        }
        self.num_sources = len(self.sources)

    def __len__(self):
        """
        Returns the total number of sources.
        """
        return self.num_sources # len(self.target_data)

    def __getitem__(self, idx):
        """
        Returns a batch of source data, target data, and source labels with metadata
        """
        # Get a random source (exclude target_source)
        source_name = list(self.sources.keys())[idx] # comment it out

        # Sample from the source
        source_data = self.sources[source_name]
        source_X = (
            source_data.obsm["X_pca"] if self.use_pca else source_data.X
        )
        source_moas = source_data.obs.Metadata_moa
        source_cpds = source_data.obs.Metadata_JCP2022
        source_microscope = source_data.obs.Metadata_Microscope_Name

        # Randomly sample from source data
        source_indices = np.random.choice(len(source_X), self.batch_size, replace=False)
        source_batch = torch.tensor(source_X[source_indices], dtype=torch.float32)
        source_moa_batch = [
            source_moas.iloc[i] if not pd.isna(source_moas.iloc[i]) else "NA" for i in source_indices
        ]
        source_cpd_batch = [
            source_cpds.iloc[i] if not pd.isna(source_cpds.iloc[i]) else "NA" for i in source_indices
        ]
        source_microscope_batch = [
            source_microscope.iloc[i] if not pd.isna(source_microscope.iloc[i]) else "NA" for i in source_indices
        ]

        # Randomly sample from target data
        target_indices = np.random.choice(len(self.target_X), self.batch_size, replace=False)
        target_batch = torch.tensor(self.target_X[target_indices], dtype=torch.float32)
        target_moa_batch = [
            self.target_moas.iloc[i] if not pd.isna(self.target_moas.iloc[i]) else "NA" for i in target_indices
        ]
        target_cpd_batch = [
            self.target_cpds.iloc[i] if not pd.isna(self.target_cpds.iloc[i]) else "NA" for i in target_indices
        ]
        target_microscope_batch = [
            self.target_microscope.iloc[i] if not pd.isna(self.target_microscope.iloc[i]) else "NA" for i in target_indices
        ]

        # One-hot encode the source label
        source_label = torch.eye(self.num_sources)[idx]

        return {
            "source": {
                "x": source_batch,
                "moa": source_moa_batch,
                "cpd": source_cpd_batch,
                "microscope": source_microscope_batch,
            },
            "target": {
                "x": target_batch,
                "moa": target_moa_batch,
                "cpd": target_cpd_batch,
                "microscope": target_microscope_batch,
            },
            "source_label": source_label,
        }
    
def create_training_dataloader(adata, batch_size=64, exclude_source="source_2", use_pca=True):
    """
    Create a dataloader for training by pooling all sources except the target.
    """
    dataset = PooledDataset(adata, target_source=exclude_source, batch_size=batch_size, use_pca=use_pca) # remov batch size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # shall I set batch size to 1 here??

    return dataloader
