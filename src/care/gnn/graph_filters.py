"""
Module containing a set of filter functions for graphs in the Geometric PyTorch format.
These filters are applied before the inclusion of the graphs in the HetGraphDataset objects.
"""

import torch
from torch_geometric.utils import to_networkx
from ase import Atoms
from torch_geometric.data import Data
from torch import where, tensor
from networkx import is_connected, cycle_basis


def fragment_filter(graph: Data, 
                    adsorbate_elems: list[str]) -> bool:
    """Check adsorbate fragmentation in the graph.
    Args:
        graph(Data): Adsorption graph.
    Returns:
        (bool): True = Adsorbate not fragmented in the graph
                False = Adsorbate fragmented in the graph
    """
    assert graph.x is not None, "x should not be None"
    assert graph.num_nodes is not None, "num_nodes should not be None"
    adsorbate = extract_adsorbate(graph, adsorbate_elems)
    graph_nx = to_networkx(adsorbate, to_undirected=True)
    if adsorbate.num_nodes != 1 and adsorbate.num_edges != 0:
        if is_connected(graph_nx):
            return True
        else:
            print(f"{graph.formula}: Fragmented adsorbate.\n".format(
                graph.formula))
            return False
    else:
        return True
    

def extract_adsorbate(graph: Data,
                     adsorbate_elems: list[str]) -> bool:
    """Extract adsorbate from the graph."""
    adsorbate_elems_idxs = [
        graph.node_feats.index(elem) for elem in adsorbate_elems
    ]
    adsorbate_nodes = []
    for node_idx in range(graph.num_nodes):
        idx = where(graph.x[node_idx, :] == 1)[0][0].item()
        if idx in adsorbate_elems_idxs:
            adsorbate_nodes.append(node_idx)
    return graph.subgraph(tensor(adsorbate_nodes))

def is_ring(graph: Data, 
            adsorbate_elems) -> bool:
    """Check if the graph contains a ring."""
    adsorbate = extract_adsorbate(graph, adsorbate_elems)
    graph_nx = to_networkx(adsorbate, to_undirected=True)
    cycles = list(cycle_basis(graph_nx))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    if len(ring_nodes) > 0:
        print(f"{graph.formula}: Ring detected.\n".format(graph.formula))
        return True
    else:
        return False

    
def H_filter(graph: Data, 
             adsorbate_elems: list[str]) -> bool:
    """
    Graph filter that checks the connectivity of H atoms whithin the adsorbate.
    Each H atoms must be connected to maximum one atom within the adsorbate.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        adsorbate_elems(list[str]): List of atomic elements in the adsorbate
    Returns:
        (bool): True = Correct connectivity for all H atoms in the adsorbate
                False = Bad connectivity for at least one H atom in the adsorbate
    """
    # 1) Get indices of adsorbate atoms in the one-hot encoder
    adsorbate_elems_indices = [graph.node_feats.index(element) for element in adsorbate_elems]  
    H_index = graph.node_feats.index("H")
    # 2) Find indices of hydrogen (H) nodes in the graph
    H_nodes_indexes = []
    for i in range(graph.num_nodes):
        if graph.x[i, H_index] == 1:
            H_nodes_indexes.append(i)
    # 3) Apply filter to H nodes (Just one bad connected H makes the whole graph wrong)
    for node_index in H_nodes_indexes:
        counter = 0  # edges between H and atoms in the adsorbate
        for j in range(graph.num_edges):
            if node_index == graph.edge_index[0, j]:  # NB: in PyG each edge repeated twice to have undirected graph
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0][0].item()  
                counter += 1 if other_atom in adsorbate_elems_indices else 0
        if counter > 1: 
            print("H connectivity filter failed for {}".format(graph.formula))
            return False
    return True

def C_filter(graph: Data, 
             adsorbate_elems: list[str]) -> bool:
    """
    Graph filter that checks the connectivity of C atoms whithin the adsorbate.
    Each C atom must be connected to maximum 4 atoms within the molecule.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        adsorbate_elems(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Correct connectivity for all C atoms in the molecule
                False = Bad connectivity for at least one C atom in the molecule
    """
    # 1) Get indices of molecule atoms in the one-hot encoder
    adsorbate_elems_indices = [graph.node_feats.index(element) for element in adsorbate_elems]   
    C_index = adsorbate_elems_indices[graph.node_feats.index("C")]
    # 2) Find indices of carbon (C) nodes in the graph
    C_nodes_indices = [index for index in range(graph.num_nodes) if graph.x[index, C_index] == 1]
    # 3) Apply filter to C nodes (Just one bad connected C makes the whole graph wrong)
    for node_index in C_nodes_indices:
        counter = 0  # number of edges between C and atoms belonging to molecule
        for j in range(graph.num_edges):
            if node_index == graph.edge_index[0, j]:  # NB: in PyG each edge repeated twice in order to have undirected graph
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0][0].item()  
                counter += 1 if other_atom in adsorbate_elems_indices else 0
        if counter > 4: 
            print("C connectivity filter failed for {}".format(graph.formula))
            return False
    return True

    
def adsorption_filter(graph: Data,  
                      adsorbate_elems: list[str]) -> bool:
    """
    Check presence of metal atoms in the adsorption graphs.
    sufficiency condition: if there is at least one atom different from C, H, O, N, S, 
    then the graph is considered as an adsorption graph.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        adsorbate_elems(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Metal catalyst present in the adsorption graph
                False = No metal catalyst in the adsorption graph
    """
    adsorbate_elems_indices = [graph.node_feats.index(element) for element in adsorbate_elems]
    for node_index in range(graph.num_nodes):
        index = torch.where(graph.x[node_index, :] == 1)[0][0].item()
        if index not in adsorbate_elems_indices:
            return True
    print(f"{graph.formula}: No surface representation")
    return False
    

def ase_adsorption_filter(atoms: Atoms,
                          adsorbate_elems: list[str]) -> bool:
    """
    Check that the adsorbate has not been incorporated in the bulk.

    Args:
        graph (Data): Input adsorption/molecular graph.
        adsorbate_elems (list[str]): List of atomic elements in the molecule
    
    Returns:
        (bool): True = Adsorbate is not incorporated in the bulk
                False = Adsorbate is incorporated in the bulk
    """
    min_adsorbate_z = min([atom.position[2] for atom in atoms if atom.symbol in adsorbate_elems])
    max_surface_z = max([atom.position[2] for atom in atoms if atom.symbol not in adsorbate_elems])
    if min_adsorbate_z < 0.8 * max_surface_z:
        print(f"{atoms.get_chemical_formula(mode='metal')}: Adsorbate incorporated in the bulk.")
        return False
    else:
        return True
