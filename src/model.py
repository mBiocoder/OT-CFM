from torchcfm.models import MLP

def create_ot_cfm_model(adata, use_pca = True, time_varying=True, w=64):
    """Create the model and check the dim.."""
    #check dim
    if use_pca:
      dim = adata.obsm['X_pca'].shape[1]
    else:
      dim = adata.X.shape[1]
    return MLP(dim=dim, time_varying=time_varying, w=w)