# Scripts directory

This folder contains the end-user scripts for creating and testing the graph neural networks.
To run them, activate first the environment via `conda activate GNN`. For checking the arguments needed for running the scripts, type `python filename.py -h`. 

- `train_GNN.py` : Train a GNN with the desired hyperparameter settings defined in a .toml input file. You will find a demo video in the `media` directory. You will find some template file in the folder `input_train_GNN`. NB: Before running the script with the provided templates, check the input .toml file and change the data path to your FG_dataset folder location (last line of the .toml file)! 
- `hypopt_GNN.py`: Perform a hyperparameter optimization with RayTune using the ASHA algorithm. The hyperparameter space must be defined within the python script.
- `GNNvsDFT.py`: Compare GNN prediction to a DFT sample you provide. You will find one VASP calculation example in the folder `input_GNNvsDFT`.
- `interactive_graph_creator.py`: GNN interface for drawing the adsorption/gas graphs and get the GNN energy and adsorption energy. You will find a demo video in the `media` directory.

