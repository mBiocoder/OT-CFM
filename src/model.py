from torchcfm.models import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_ot_cfm_model(adata, use_pca=True, time_varying=True, w=64, num_sources=None):
    """ Create the OT-CFM model and check the dimension of the input, accounting for the one-hot encoded source. """
    
    # Check dimensionality based on whether PCA is used
    if use_pca:
        dim = adata.obsm['X_pca'].shape[1]
    else:
        dim = adata.X.shape[1]

    if num_sources is None:
        num_sources = len(adata.obs['Metadata_Source'].unique()) - 1
    
    # Dims 
    input_dim = dim + num_sources  # The input features + the one-hot encoding of the source
    
    # Adjusted input_dim for time if time_varying
    if time_varying:
        input_dim += 1  # Add 1 for the time dimension
    
    # Return MMLP model
    return MLP(dim=input_dim, time_varying=time_varying, w=w)

    
class MLP(nn.Module):
    def __init__(self, dim, time_varying=True, w=64):
        super(MLP, self).__init__()
        self.dim = dim 
        self.time_varying = time_varying
        self.w = w

        # Define layers with dynamic input dimension
        self.layer1 = nn.Linear(self.dim, self.w)  # Dynamic input size based on dim!!!
        self.layer2 = nn.Linear(self.w, self.w)
        self.layer3 = nn.Linear(self.w, 1)

    def forward(self, source_batch, source_one_hot, time):
        source_one_hot = source_one_hot.unsqueeze(1).repeat(1, source_batch.shape[1], 1)
        source_batch = source_batch.unsqueeze(1).expand(-1, source_one_hot.shape[1], -1)

        # Debugging
        #print("source_batch shape:", source_batch.shape)  # [1, 50, 50]
        #print("source_one_hot shape:", source_one_hot.shape)  # [1, 50, 9]
        #print("time shape:", time.shape)  # [1, 50, 1]

        # Concatenate inputs along the feature dimension
        x = torch.cat([source_batch, source_one_hot, time], dim=-1)  
        #7print("Shape of concatenated input:", x.shape) [1, 50, 60]

        # Pass through layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x









