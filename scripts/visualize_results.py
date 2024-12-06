import scanpy as sc
from ot_cfm.transport import load_ot_cfm_model, transport, create_dataloader

# Load dataset and init model
target2_moa = sc.read_h5ad("../../data/Tim_target2_wellres_featuresimputed_druginfoadded_pycytominer.h5ad")
ot_cfm_model = create_ot_cfm_model(adata=target2_moa, use_pca=True)
ot_cfm_optimizer = create_ot_cfm_optimizer(ot_cfm_model)

translated_sources = []

# Load and transport sources
for source in target2_moa.obs["Metadata_Source"].unique():
    if source == "source_2":
        continue

    dataloader = create_dataloader(target2_moa, source, batch_size=64, use_pca=True)
    load_ot_cfm_model(ot_cfm_model, ot_cfm_optimizer, f"./saved_models/ot_cfm_{source}_to_source_2.pt")
    translated_source_adata = transport(dataloader, ot_cfm_model, x1_label=source, device="cpu")
    translated_sources.append(translated_source_adata)

# Combine all sources and plot PCA
all_sources_adata = sc.concat(translated_sources, axis=0)
source_2_data = target2_moa[target2_moa.obs["Metadata_Source"] == "source_2"].copy()
all_sources_adata = sc.concat([all_sources_adata, source_2_data], join="outer")

# PCA
sc.pl.pca(all_sources_adata, color="Metadata_Source")

# UMAP
sc.pp.neighbors(all_sources_adata, use_rep="X_pca")
sc.tl.umap(all_sources_adata)
sc.pl.umap(all_sources_adata, color="Metadata_Source")
