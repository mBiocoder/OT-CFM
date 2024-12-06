# Batch correction using optimal transport conditional flow matching (OT-CFM)

## Repo structure

src/ot_cfm/: Core package implementation.
* data_utils.py: Handles data loading and preprocessing.
* model.py: Defines the OT-CFM model.
* training.py: Manages training and model saving.
* transport.py: Includes transport logic and visualization utilities.

scripts/: Contains runnable scripts for workflows and visualizations.

data/: Stores input data 

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

