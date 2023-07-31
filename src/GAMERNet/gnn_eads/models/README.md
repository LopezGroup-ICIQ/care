# Models directory

This directory stores all the GNN models that are created when you perform a GNN model training with the script `train_GNN.py`.

Each model folder contains at least the following files:

1. `model.pth`: it stores the GNN model architecture.
2. `GNN.pth`: it stores the model parameters.
3. `performance.txt`: summary of the performed model training.

In order to load each model, use the `PreTrainedModel` class present in `gnn_eads.nets` module (See `notebooks` directory for details).
