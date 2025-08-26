"""
Module containing a set of filter functions for graphs in the Geometric PyTorch format.
These filters are applied before the inclusion of the graphs in the HetGraphDataset objects.
"""

from ase import Atoms
from networkx import cycle_basis, is_connected
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def fragment_filter(graph: Data, adsorbate_elems: list[str]) -> bool:
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
            return False
    else:
        return True


def extract_adsorbate(graph: Data, adsorbate_elems: list[str]) -> bool:
    """Extract adsorbate from the graph."""
    adsorbate_nodes = [
        node_idx
        for node_idx in range(graph.num_nodes)
        if graph.elem[node_idx] in adsorbate_elems
    ]
    return graph.subgraph(tensor(adsorbate_nodes))


def is_ring(graph: Data, adsorbate_elems: list[str]) -> bool:
    """Check if the graph contains a ring."""
    adsorbate = extract_adsorbate(graph, adsorbate_elems)
    graph_nx = to_networkx(adsorbate, to_undirected=True)
    cycles = list(cycle_basis(graph_nx))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    if len(ring_nodes) > 0:
        return True
    else:
        return False


def H_filter(graph: Data, adsorbate_elems: list[str]) -> bool:
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
    H_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "H"]
    for node_index in H_nodes_indices:
        counter = 0  # edges between H and atoms in the adsorbate
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if graph.elem[graph.edge_index[1, j]] in adsorbate_elems else 0
        if counter > 1:
            return False
    return True


def C_filter(graph: Data, adsorbate_elems: list[str]) -> bool:
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
    C_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "C"]
    for node_index in C_nodes_indices:
        counter = 0  # number of edges between C and atoms belonging to molecule
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if graph.elem[graph.edge_index[1, j]] in adsorbate_elems else 0
        if counter > 4:
            return False
    return True


def adsorption_filter(graph: Data, adsorbate_elems: list[str]) -> bool:
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
    if graph.metal == "N/A" and graph.facet == "N/A":
        return True
    else:
        return False if all([elem in adsorbate_elems for elem in graph.elem]) else True


def ase_adsorption_filter(atoms: Atoms, adsorbate_elems: list[str]) -> bool:
    """
    Check that the adsorbate has not been incorporated in the bulk.

    Args:
        graph (Data): Input adsorption/molecular graph.
        adsorbate_elems (list[str]): List of atomic elements in the molecule

    Returns:
        (bool): True = Adsorbate is not incorporated in the bulk
                False = Adsorbate is incorporated in the bulk
    """
    if all([atom.symbol in adsorbate_elems for atom in atoms]):
        return True
    min_adsorbate_z = min(
        [atom.position[2] for atom in atoms if atom.symbol in adsorbate_elems]
    )
    max_surface_z = max(
        [atom.position[2] for atom in atoms if atom.symbol not in adsorbate_elems]
    )
    if min_adsorbate_z < 0.8 * max_surface_z:
        print(
            f"{atoms.get_chemical_formula(mode='metal')}: Adsorbate incorporated in the bulk."
        )
        return False
    else:
        return True
