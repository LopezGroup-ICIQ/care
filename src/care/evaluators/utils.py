"""
This module contains functions and classes for creating, manipulating and analyzing graphs
from ASE Atoms objects to PyG Data format. Readapted from GAME-Net-UQ, but general for any structure.
"""

from itertools import product

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers
from networkx import (
    Graph,
    cycle_basis,
    get_node_attributes,
    is_connected,
    set_node_attributes,
    draw_networkx, 
    kamada_kawai_layout
)
from scipy.spatial import Voronoi
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from care.constants import CORDERO, RGB_COLORS
import matplotlib.pyplot as plt


def get_voronoi_neighbourlist(
    atoms: Atoms, tol: float, scaling_factor: float, atom_tags: list[int]
) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The surface does not contain elements present in the adsorbate.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tol (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
        atom_tags (list[int]): tags defining whether an atom is part of the adsorbate or the surface.
                              1 for adsorbate, 0 for surface.

    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.

    Notes:
        Each connection is represented once, i.e. if atom A is connected to atom B, the pair (A, B) will be present in the list,
        but not the pair (B, A).
    """

    # First necessary condition for two atoms to be linked: Sharing a Voronoi facet
    coords_arr = np.repeat(
        np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0
    )
    mirrors = np.repeat(
        np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1),
        coords_arr.shape[1],
        axis=1,
    )
    corrected_coords = np.reshape(
        coords_arr + mirrors,
        (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]),
    )
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(
        pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0
    )
    increment = 0.0
    pairs = []
    while True:
        pairs.clear()
        for i, j in pairs_corr:
            a1, a2 = atoms[i].symbol, atoms[j].symbol
            d = atoms.get_distance(i, j, mic=True)

            # base threshold
            thresh = CORDERO[a1] + CORDERO[a2] + tol

            # scaling correction if crossing adsorbate ↔ substrate boundary
            if atom_tags[i] != atom_tags[j]:
                corr = max(scaling_factor + increment - 1.0, 0)
                if atom_tags[i] == 1:  # i is adsorbate
                    thresh += corr * CORDERO[a2]
                else:  # j is adsorbate
                    thresh += corr * CORDERO[a1]

            if d <= thresh:
                pairs.append((i, j))
        if any(atom_tags[i] != atom_tags[j] for i, j in pairs):
            return np.sort(np.array(pairs, dtype=int), axis=1)
        increment += 0.2


def atoms_to_nx(
    atoms: Atoms,
    voronoi_tolerance: float,
    scaling_factor: float,
    surface_order: int,
    atom_tags: list[int],
) -> Graph:
    """
    Convert ASE Atoms object to NetworkX graph, representing the adsorbate-surface system.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        voronoi_tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the surface atoms.
        surface_order (int): order of the surface neighbours to be included in the graph. If set to -1,
                            all surface slab is included.
        atom_tags (list[int]): tags defining whether an atom is part of the adsorbate or the surface.
    Returns:
        Graph: NetworkX graph representing the adsorbate-metal system.
    """
    neighbour_list = get_voronoi_neighbourlist(
        atoms, voronoi_tolerance, scaling_factor, atom_tags
    )
    adsorption_ensemble = {atom.index for atom in atoms if atom_tags[atom.index] == 1}
    surf_hops = {0: list(adsorption_ensemble)}
    if surface_order == -1:
        surface_order = 100    
    for _ in range(surface_order):
        surface_ensemble = {
            pair[1] if pair[0] in adsorption_ensemble else pair[0]
            for pair in neighbour_list
            if (pair[0] in adsorption_ensemble and pair[1] not in adsorption_ensemble)
            or (pair[1] in adsorption_ensemble and pair[0] not in adsorption_ensemble)
        }
        surf_hops[_ + 1] = list(surface_ensemble)
        adsorption_ensemble = adsorption_ensemble.union(surface_ensemble)
        if len(adsorption_ensemble) == len(atoms):
            break
    graph = Graph()
    graph.add_nodes_from(list(adsorption_ensemble))
    set_node_attributes(graph, {i: atoms[i].symbol for i in graph.nodes()}, "elem")
    ensemble_neighbour_list = [
        pair
        for pair in neighbour_list
        if pair[0] in graph.nodes() and pair[1] in graph.nodes()
    ]
    graph.add_edges_from(ensemble_neighbour_list)
    return graph, surf_hops


def atoms_to_data(
    structure: Atoms, 
    atom_tags: list[int] = None,
    surface_order: int = -1,
    filter: bool = True,
    tol: float = 0.50
) -> Data:
    """
    Convert ASE Atoms object to PyG Data graph based on the input parameters.
    In CARE, this function is used only for intermediate species, not for transition states.
    The implementation is similar to the one in the ASE to PyG converter class, but it is not a class method and
    is used for inference. Target values are not included in the Data object.

    Args:
        structure (Atoms): ASE atoms object.
        atom_tags (list[int]): list of tags defining whether an atom is part of the adsorbate or the surface.
                               0 for surface atoms, 1 for adsorbate atoms. If not provided, the function tries to extract this info 
                               from the ASE input structure metadata
        surface_order (int): order of the surface neighbours to be included in the graph. If set to -1,
                            all surface slab is included.
        filter (bool): whether to apply connectivity checks on final graph.
    Returns:
        graph (Data): PyG Data object.
    """
    if atom_tags is None or len(atom_tags) == 0:
        if "atom_tags" in structure.arrays:
            atom_tags = structure.get_array("atom_tags").tolist()
        else:
            raise ValueError(
                "No atom_tags provided and ASE structure has no 'atom_tags' array"
            )
    nx, surf_hops = atoms_to_nx(
            structure, tol, 1.25, surface_order, atom_tags
    )
    elem_list = list(get_node_attributes(nx, "elem").values())
    idx_list = list(get_node_attributes(nx, "elem").keys())
    elem_enc = np.array([atomic_numbers[symbol] for symbol in elem_list]).reshape(-1, 1)
    x = torch.from_numpy(elem_enc).float()
    nodes_list = list(nx.nodes)
    edge_tails_heads = [
        (nodes_list.index(edge[0]), nodes_list.index(edge[1])) for edge in nx.edges
    ]
    edge_tails = [x for x, _ in edge_tails_heads] + [y for _, y in edge_tails_heads]
    edge_heads = [y for _, y in edge_tails_heads] + [x for x, _ in edge_tails_heads]
    edge_index = torch.tensor([edge_tails, edge_heads], dtype=torch.long)
    graph = Data(x, 
                 edge_index, 
                 elem=elem_list, 
                 idx=idx_list, 
                #  surf_hops=surf_hops, 
                 formula=structure.get_chemical_formula())

    # CONNECTIVITY CHECKS
    if filter:
        if not H_filter(graph, atom_tags):
            return None
        if not C_filter(graph, atom_tags):
            return None
        if is_adsorbate_fragmented(graph, atom_tags):
            return None
    return graph


def extract_adsorbate(graph: Data, atom_tags: list[int]) -> Data:
    """Extract adsorbate from the graph."""
    adsorbate_nodes = [
        node_idx
        for node_idx in range(graph.num_nodes)
        if atom_tags[graph.idx[node_idx]] == 1
    ]
    return graph.subgraph(tensor(adsorbate_nodes))


def is_adsorbate_fragmented(graph: Data, atom_tags: list[int]) -> bool:
    """Check adsorbate fragmentation in the graph.
    Args:
        graph(Data): Adsorption graph.
        atom_tags (list[int]): list of tags defining whether an atom is part of the adsorbate or the surface.
    Returns:
        (bool): True = Fragmented adsorbate
                False = Connected adsorbate
    """
    adsorbate = extract_adsorbate(graph, atom_tags)
    graph_nx = to_networkx(adsorbate, to_undirected=True, remove_self_loops=True)
    if adsorbate.num_nodes == 1 and adsorbate.num_edges == 0:
        return False
    return not is_connected(graph_nx)


def is_ring(graph: Data, atom_tags: list[int]) -> bool:
    """Check if the graph contains a ring."""
    adsorbate = extract_adsorbate(graph, atom_tags)
    graph_nx = to_networkx(adsorbate, to_undirected=True, remove_self_loops=True)
    cycles = list(cycle_basis(graph_nx))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    if len(ring_nodes) > 0:
        return True
    else:
        return False


def H_filter(graph: Data, atom_tags: list[int]) -> bool:
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
    H_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "H" and atom_tags[graph.idx[i]] == 1]
    for node_index in H_nodes_indices:
        counter = 0  # bonds between H and other adsorbate atoms
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if atom_tags[graph.idx[graph.edge_index[1, j]]] == 1 else 0
        if counter > 1:
            return False
    return True


def C_filter(graph: Data, atom_tags: list[int]) -> bool:
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
    C_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "C" and atom_tags[graph.idx[i]] == 1]
    for node_index in C_nodes_indices:
        counter = 0  # nbonds between C and other adsorbate atoms
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if atom_tags[graph.idx[graph.edge_index[1, j]]] == 1 else 0
        if counter > 4:
            return False
    return True


def adsorption_filter(graph: Data, atom_tags: list[int]) -> bool:
    """
    Check presence of surface atoms in the adsorption graph.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        atom_tags (list[int]): List of tags defining whether an atom is part of the adsorbate or the surface
    Returns:
        (bool): True = Surface atoms present in the adsorption graph
                False = No surface atoms in the adsorption graph
    """
    return False if all([atom_tags[graph.idx[i]] == 1 for i in range(graph.num_nodes)]) else True


def ase_adsorption_filter(atoms: Atoms, atom_tags: list[int]) -> bool:
    """
    Check that the adsorbate has not been incorporated in the bulk.

    Args:
        graph (Data): Input adsorption/molecular graph.
        atom_tags (list[int]): List of tags defining whether an atom is part of the adsorbate or the surface

    Returns:
        (bool): True = Adsorbate is not incorporated in the bulk
                False = Adsorbate is incorporated in the bulk
    """
    if all([atom_tags[i] == 1 for i in range(len(atom_tags))]):
        return True
    min_adsorbate_z = min(
        [atom.position[2] for atom in atoms if atom_tags[atom.index] == 1]
    )
    max_surface_z = max(
        [atom.position[2] for atom in atoms if atom_tags[atom.index] == 0]
    )
    if min_adsorbate_z < 0.8 * max_surface_z:
        return False
    else:
        return True
    

def pyg_to_nx(graph: Data) -> Graph:
    """
    Convert graph in pytorch_geometric to NetworkX type.
    For each node in the graph, the label corresponding to the atomic species
    is added as attribute together with a corresponding color.
    Args:
        graph(torch_geometric.data.Data): torch_geometric graph object.
    Returns:
        nx_graph(networkx.classes.graph.Graph): NetworkX graph object.
    """
    n_nodes = graph.num_nodes
    atom_list = [graph.elem[i] for i in range(n_nodes)]
    g = to_networkx(graph, to_undirected=True)
    connections = list(g.edges)
    nx_graph = Graph()
    for i in range(n_nodes):
        nx_graph.add_node(i, elem=atom_list[i], rgb=RGB_COLORS[atom_list[i]])
    nx_graph.add_edges_from(connections)
    return nx_graph


def nx_to_pyg(graph_nx: Graph) -> Data:
    """
    Convert graph object from networkx to pytorch_geometric type.
    Args:
        graph(networkx.classes.graph.Graph): networkx graph object
    Returns:
        new_g(torch_geometric.data.Data): torch_geometric graph object
    """
    n_nodes = graph_nx.number_of_nodes()
    n_edges = graph_nx.number_of_edges()
    node_features = torch.zeros((n_nodes, 1))
    edge_features = torch.zeros((n_edges, 1))
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    node_index = torch.zeros((n_nodes), dtype=torch.long)
    for i, node in enumerate(graph_nx.nodes):
        node_index[i] = node
        node_features[i, 0] = graph_nx.nodes[node]["elem"]
    for i, edge in enumerate(graph_nx.edges):
        edge_index[0, i] = edge[0]
        edge_index[1, i] = edge[1]

    graph_pyg = Data(
        x=node_features, edge_index=edge_index, y=node_index
    )
    return graph_pyg


def graph_plotter(
    graph: Data,
    node_size: int = 320,
    font_color: str = "white",
    font_weight: str = "bold",
    alpha: float = 1.0,
    arrowsize: int = 10,
    width: float = 1.2,
    dpi: int = 200,
    figsize: tuple[int, int] = (4, 4),
    node_index: bool = True,
    text: str = None,
):
    """
    Visualize graph with atom labels and colors. Working also for TSs.
    Kamada_kawai_layout engine gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    nx_graph = pyg_to_nx(graph)
    labels = get_node_attributes(nx_graph, "elem")
    colors = list(get_node_attributes(nx_graph, "rgb").values())
    edge_colors = ["black" for edge in nx_graph.edges]
    plt.figure(figsize=figsize, dpi=dpi)
    draw_networkx(
        nx_graph,
        labels=labels,
        node_size=node_size,
        font_color=font_color,
        font_weight=font_weight,
        node_color=colors,
        edge_color=edge_colors,
        alpha=alpha,
        arrowsize=arrowsize,
        width=width,
        pos=kamada_kawai_layout(nx_graph),
        linewidths=0.5,
    )
    if node_index:
        pos_dict = kamada_kawai_layout(nx_graph)
        for node in nx_graph.nodes:
            x, y = pos_dict[node]
            plt.text(x + 0.05, y + 0.05, node, fontsize=7)
    if text != None:
        plt.text(0.03, 0.9, text, fontsize=10)
    plt.axis("off")
    plt.draw()

def connectivity_signature(nx_g):
    """Return a sorted list of (element, sorted neighbor elements) for each node."""
    sig = []
    for n in nx_g.nodes():
        elem = nx_g.nodes[n]['elem']
        neighbor_elems = sorted([nx_g.nodes[neigh]['elem'] for neigh in nx_g.neighbors(n)])
        sig.append((elem, tuple(neighbor_elems)))
    return sorted(sig)
