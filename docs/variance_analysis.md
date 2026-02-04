# Decision-axis vs orthogonal variance

This experiment studies how within-class variance evolves during self-supervised training, separating variance along discriminative directions (decision axes) from variance in task-irrelevant orthogonal subspaces. 

The goal is to show that, across SSL methods, variance collapses rapidly along decision axes, while substantial variance persists in orthogonal directions.

## Experiment setup

For a given pretrained SSL model and training epoch:
1. Select class pairs. This can be either random selection or based on direction CDNV scores.
2. For each class pair:
   - Collect representations of samples from both classes.
   - Compute the decision axis as the vector connecting the class centroids.
   - Project representations onto the decision axis and its orthogonal complement.
   - Calculate variance along the decision axis and orthogonal directions.

## How to run the experiment
 
 **Prerequisites**:
 - Pretrained SSL checkpoints saved at multiple epochs.
 - Dataset config stored in the checkpoint. (used to rebuild the datamodule)

 Run the following command:

 ```python
 python training_scratch/variance_bar_plots.py \
  --ckpt_dir </path/to/checkpoints> \
  --out_csv </path/to/output.csv> \
  --start 0 \
  --end 1000 \
```

This will generate a CSV summary of pairwise variance metrics across epochs.

## Reproducing Figure 4

To generate the plots shown in Figure 3 of the paper, use the [variance_plot.ipynb](../notebooks/variance_plot.ipynb) notebook. Below is an example plot produced by the notebook:

<img src="../git_figures/variance_decomp_mae.png" width="600">