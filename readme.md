# Structural Privacy in GNNs

## Requirements

This code is implemented in Python 3.9, and relies on the following packages:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.8.1
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.7.0
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.2.4
- [Numpy](https://numpy.org/install/) >= 1.20.2
- [Seaborn](https://seaborn.pydata.org/) >= 0.11.1  

## Usage

### Replicating the paper's results
In order to replicate our experiments and reproduce the paper's results, you must do the following steps:  
1. Run ``python experiments.py -n test create --top_k_based --threshold_based ``
2. Run ``python experiments.py -n test exec --all``
   All the datasets will be downloaded automatically into ``datasets`` folder, and the results will be stored in ``results`` directory.
3. Go through ``plot_k_rr.ipynb`` and ``plot_wandb.ipynb`` notebooks to visualize the results.

