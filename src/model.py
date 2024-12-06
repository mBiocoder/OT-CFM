from torchcfm.models import MLP

def create_ot_cfm_model(adata, use_pca=True, time_varying=True, w=64, num_sources=None):
    """ Create the OT-CFM model and check the dimension of the input, accounting for the one-hot encoded source.
    """
    # Check dimensionality based on whether PCA is used
    if use_pca:
        dim = adata.obsm['X_pca'].shape[1]
    else:
        dim = adata.X.shape[1]

    if num_sources is None:
        num_sources = len(adata.obs['Metadata_Source'].unique())
    
    # Dims 
    input_dim = dim + num_sources  # The input features + the one-hot encoding of the source
    
    # Return MMLP model
    return MLP(dim=input_dim, time_varying=time_varying, w=w)


class MLP(nn.Module):
    def __init__(self, dim, time_varying=True, w=64):
        super(MLP, self).__init__()
        self.dim = dim
        self.time_varying = time_varying
        self.w = w
        
        # Not sure about MLP setup here (!!!)
        self.layer1 = nn.Linear(self.dim, self.w)
        self.layer2 = nn.Linear(self.w, self.w)
        self.layer3 = nn.Linear(self.w, 1)  
        
    def forward(self, source_batch, source_one_hot):
        # Concatt feature vector and the one-hot encoded source vector
        x = torch.cat([source_batch, source_one_hot], dim=-1)  
        
        # Pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x) 
        return x
