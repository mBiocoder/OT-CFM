# Batch correction using optimal transport conditional flow matching (OT-CFM)

## Repo structure

src/: Core package implementation
* data_utils.py: Handles data loading and preprocessing.
* model.py: Defines the OT-CFM model.
* training.py: Manages training and model saving.
* transport.py: Includes transport logic and visualization utilities.

scripts/: Includes standalone scripts for running and visualizing OT-CFM 
* run_ot_cfm.py:  Trains and saves models for all sources except source_2 on TARGET2 JUMP dataset.
* visualize_results.py: Loads translated sources and visualizes them using PCA and UMAP.

notebooks/: 
* example_run_ot_cfm.ipynb: Demonstrates how to run the OT-CFM pipeline.

## Workflow
Optimal Transport with Conditional Flow Matching (OT-CFM) aligns data distributions between different sources or batches. By combining optimal transport (OT) and a neural network-based flow matching model, OT-CFM learns the transformation dynamics betweeen source and target domains. The OT computes the coupling matrix to nicely match points from the source to the target domain. The NN training predicts transformation velocities between distributions, then Neural ODE simmulates transformation trajectories for new data points.

### Input
Single-cell AnnData with PCA features and metadata

##### Usage: 
* Use time_varying=True for tasks requiring gradual or interpretable transformations over time
* Use time_varying=False for faster training and when smooth trajectories are unnecessary
* Confirm PCA pre-processing compatibility if use_pca=True

### Model Training
Train the OT-CFM model to align distributions via batch-wise training
Compute loss and update the model weights

### Transformation
Use the trained model and Neural ODE to transport pooled source data into the target domain
Save and visualize the transformed data (e.g., PCA/UMAP plots)

### Output
Transformed AnnData object with aligned data across sources
Visualization of the transformation results
Saved OT-CFM model for reuse

## Pooled batch mapping:
* Sampling: Pooled data from all sources except source_2 (in our example this is the target). Then we sample batches from a random source and source_2.
* Training: Update to pass one-hot encoded vectors of the source as inputs to the model.
* Correction: Map all sources into source_2
