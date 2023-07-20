"""Module containing functions for converting ASE Atoms objects to the graph representation used by the selected GNN model."""

from ase import Atoms
from torch_geometric.data import Data
from torch import tensor

def build_graph_skeleton(structure: Atoms,
                         structure_params: dict) -> Data:
    """Given an ASE Atoms object, build the skeleton of the PyTorch Geometric Data object.
    """
    pass

def featurize_nodes(structure: Atoms,
                    pyg_graph: Data,
                    node_features: dict) -> Data:
    """Given an ASE Atoms object, featurize the nodes of the PyTorch Geometric Data object.
    """
    # Select items from node_features dictionary whose values are True
    node_features = {k: v for k, v in node_features.items() if v}
    # Initialize node features tensor with shape (n_nodes, n_features)
    x = pyg_graph.x
    pass

def atoms_to_pyg(structure: Atoms, 
                 conversion_params: dict) -> Data:
    """Given an ASE Atoms object, convert it to a PyTorch Geometric Data object.
    
    Args:
        structure (Atoms): ASE Atoms object to convert.
        conversion_params (dict): Dictionary of conversion parameters.
        
    Returns:
        Data: PyTorch Geometric Data object.
    """
    
    structure_params = conversion_params['structure']
    node_features = conversion_params['features']
    pyg_graph = build_graph_skeleton(structure, structure_params)
    x = featurize_nodes(structure, pyg_graph, node_features)
    pyg_graph.x = x
    return pyg_graph