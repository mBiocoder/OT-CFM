# Batch correction using optimal transport conditional flow matching (OT-CFM)

## Repo structure

OT-CFM/
├── src/
│   ├── ot_cfm/
│   │   ├── __init__.py
│   │   ├── data_utils.py         # For data loading, preprocessing and dataloaders
│   │   ├── model.py              # contains the OT-CFM model definition
│   │   ├── training.py           # Training and saving
│   │   ├── transport.py          # Transport logic and visualization utilities
│   │   ├── ot_flow.py            # Wrapper for ExactOptimalTransportConditionalFlowMatcher
│   └── scripts/
│       ├── run_ot_cfm.py         # Main entry script for running an example workflow
│       ├── visualize_results.py  # UMAP/PCA visualization
├── data/                         
│   └── Tim_target2_wellres_featuresimputed_druginfoadded_pycytominer.h5ad
├── notebooks/                   
│   └── OT_CFM_target2_moa_without_source9.ipynb 
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py                      

## Workflow
Optimal Transport with Conditional Flow Matching (OT-CFM) aligns data distributions between different sources or batches. By combining optimal transport (OT) and a neural network-based flow matching model, OT-CFM learns the transformation dynamics betweeen source and target domains. The OT computes the coupling matrix to nicely match points from the source to the target domain. The NN training predicts transformation velocities between distributions, then Neural ODE simmulates transformation trajectories for new data points.

### Input:
* Single-cell data: AnnData object with features, metadata, and source labels
* Configuration parameters: Options for using X_PCA or not, batch size, and training epochs

### Process:
* Training: Train OT-CFM to map each source batch to the target domain (e.g., source 2 in our exmaple) and save trained models
* Translation: Use trained models to transform all source batches to align with the target and then combine transformed and original data into one dataset.
* Visualization:Use PCA/UMAP plots to assess alignment with the target domain

### Output:
* Transformed Data: New AnnData object with original and transformed data
* Visualization: PCA/UMAP plots to validate data alignment
* Saved Models: Trained OT-CFM models

