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

class PooledDataset(Dataset):
    """
    Dataset for pooling all sources except a specified target source.
    Ensures each batch consists of samples from a single source.
    """
    def __init__(self, adata, batch_size, target_source, use_pca=True):
        self.adata = adata  
        self.batch_size = batch_size
        self.target_source = target_source
        self.use_pca = use_pca
        
        # Separate target source and other sources
        self.target_data = adata[adata.obs["Metadata_Source"] == target_source].copy()
        self.other_sources = adata[adata.obs["Metadata_Source"] != target_source].copy()

        # Prepare PCA or raw features sfor target
        self.target_X = (
            self.target_data.obsm["X_pca"] if use_pca else self.target_data.X
        )
        self.target_moas = self.target_data.obs.Metadata_moa  
        self.target_cpds = self.target_data.obs.Metadata_JCP2022
        self.target_microscope = self.target_data.obs.Metadata_Microscope_Name

        # Prepare data for each other source
        self.sources = {
            source: adata[adata.obs["Metadata_Source"] == source].copy()
            for source in self.other_sources.obs["Metadata_Source"].unique()
        }
        self.num_sources = len(self.sources)

        # Initialize dictionary mapping range(num_sources) to the source id
        self.id2source = dict(zip(range(self.num_sources), 
                                  list(self.sources.keys())))

    def __len__(self):
        """Returns the total number of target samples."""
        return self.num_sources

    def __getitem__(self, source_idx):
        """
        Returns a batch of source data, target data, and source labels with metadata.
        Ensures that all source samples in the batch are from the same source.
        """
        idx_target = np.random.choice(range(len(self.target_X)),
                                     self.batch_size)  # Sample target observation index 
        # Collect a batch of targets
        target_X_batch = torch.tensor(self.target_X[idx_target], dtype=torch.float32)
        target_moas_batch = self.target_moas.iloc[idx_target].values
        target_cpds_batch = self.target_cpds.iloc[idx_target].values
        target_microscope_batch = self.target_microscope.iloc[idx_target].values
        
        # Randomly select one source for the entire batch
        source_name = self.id2source[source_idx]

        # Randomly sample the same number of observations from this source
        source_data = self.sources[source_name]
        random_idx_source = np.random.choice(len(source_data), 
                                            self.batch_size)
        source_adata_batch = source_data[random_idx_source]

        source_X_batch = (
            source_adata_batch.obsm["X_pca"] if self.use_pca else source_adata_batch.X
        )
        source_X_batch = torch.tensor(source_X_batch, dtype=torch.float32)

        # Extract source metadata
        source_moas = source_adata_batch.obs.Metadata_moa.values
        source_cpds = source_adata_batch.obs.Metadata_JCP2022.values
        source_microscope = source_adata_batch.obs.Metadata_Microscope_Name.values

        # One-hot encode the source label
        source_label = torch.eye(self.num_sources)[(source_idx*torch.ones(target_X_batch.shape[0])).long()]

        return {
            "source": {
                "x": source_X_batch,
                "moa": source_moas,
                "cpd": source_cpds,
                "microscope": source_microscope,
            },
            "target": {
                "x": target_X_batch,
                "moa": target_moas_batch,
                "cpd": target_cpds_batch,
                "microscope": target_microscope_batch,
            },
            "source_label": source_label,
        }

def collate_fn(data):
    return data[0]
    
def create_training_dataloader(adata, batch_size=64, exclude_source="source_2", use_pca=True):
    """
    Create a dataloader for training by pooling all sources except the target.
    Ensures each batch consists of observations from a single source.
    """
    dataset = PooledDataset(adata, batch_size=batch_size, target_source=exclude_source, use_pca=use_pca)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    return dataset, dataloader
