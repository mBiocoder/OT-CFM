import torch
from ot_cfm.data_utils import load_adata
from ot_cfm.model import create_ot_cfm_model, create_ot_cfm_optimizer
from ot_cfm.training import train_cfm, save_ot_cfm_model
from ot_cfm.transport import create_training_dataloader
from ot_cfm.model import ExactOptimalTransportConditionalFlowMatcher

# Load dataset
target2_moa = load_adata("../../data/Tim_target2_wellres_featuresimputed_druginfoadded_pycytominer.h5ad")

# Initialize model and optimizer
ot_cfm_model = create_ot_cfm_model(adata=target2_moa, use_pca=True)
ot_cfm_optimizer = create_ot_cfm_optimizer(ot_cfm_model)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

# Iterate and train
for source in target2_moa.obs["Metadata_Source"].unique():
    if source == "source_2":
        continue

    dataloader = create_training_dataloader(target2_moa, source, "source_2", batch_size=64, use_pca=True)
    train_cfm(ot_cfm_model, ot_cfm_optimizer, FM, dataloader, epochs=200)
    save_ot_cfm_model(ot_cfm_model, ot_cfm_optimizer, f"./saved_models/ot_cfm_{source}_to_source_2.pt")
