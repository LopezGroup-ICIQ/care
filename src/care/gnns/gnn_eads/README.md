# gnn_eads

This package contains the modules with the functions and classes used to design the Graph Neural Network (GNN) framework.

- `constants.py`: Contains variables such as the atomic radii, the chemical elements considered, one-hot encoder, etc.
- `functions.py`: All the functions used to convert DFT data to graphs, define the workflow for performing the model training, etc.
- `graph_filters.py`: Defines the filters applied to the raw graph dataset before performing the GNN training.
- `graph_tools.py`: For converting graphs from NetworkX to PyG type and visualizing graphs.
- `nets.py`: Contains the classes defining the GNN model architectures.
- `post_training.py`: Contains the function `create_model_report`, that generates a full report every time a model training is performed.
- `plot_functions.py`: The functions called by `create_model_report` for creating plots related to the training process are stored here.
- `paths.py`: For generating the paths addressing the data.
