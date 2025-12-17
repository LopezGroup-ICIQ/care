"""
This module contains functions and classes for creating, manipulating and analyzing graphs
from ASE Atoms objects to PyG Graph format. Readapted from GAME-Net-UQ, but general for any structure.
"""

from collections import defaultdict
from itertools import product

from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
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

from care.constants import CORDERO, RGB_COLORS


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


def atoms_to_data(
    atoms: Atoms,
    atom_tags: list[int] = None,
    surface_order: int = -1,
    voronoi_tolerance: float = 0.5,
    scaling_factor: float = 1.25,
    filter: bool = True,
) -> Graph:
    """
    Convert ASE Atoms object to NetworkX graph.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        voronoi_tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the surface atoms.
        surface_order (int): order of the surface neighbours to be included in the graph. If set to -1,
                            all surface slab is included.
        atom_tags (list[int]): tags defining whether an atom is part of the adsorbate or the surface.
        filter (bool): whether to apply connectivity checks on final graph.
    Returns:
        Graph: NetworkX graph representing the adsorbate-metal system.
    """
    if atom_tags is None or len(atom_tags) == 0:
        if "atom_tags" in atoms.arrays:
            atom_tags = atoms.get_array("atom_tags").tolist()
        else:
            raise ValueError(
                "No atom_tags provided and ASE structure has no 'atom_tags' array"
            )
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
    set_node_attributes(graph, {i: i for i in graph.nodes()}, "idx")
    set_node_attributes(graph, {i: atom_tags[i] for i in graph.nodes()}, "atom_tags")
    ensemble_neighbour_list = [
        pair
        for pair in neighbour_list
        if pair[0] in graph.nodes() and pair[1] in graph.nodes()
    ]
    graph.add_edges_from(ensemble_neighbour_list)
    graph.graph["surf_hops"] = surf_hops
    graph.graph["formula"] = atoms.get_chemical_formula()
    if filter:
        if not H_filter(graph):
            return None
        if not C_filter(graph):
            return None
        if is_adsorbate_fragmented(graph):
            return None
    return graph


def extract_adsorbate(graph: Graph) -> Graph:
    """Extract adsorbate from the graph."""
    adsorbate_nodes = [n for n in graph.nodes if graph.nodes[n]["atom_tags"] == 1]
    return graph.subgraph(adsorbate_nodes).copy()


def is_adsorbate_fragmented(graph: Graph) -> bool:
    """Check adsorbate fragmentation in the graph.
    Args:
        graph(Graph): Adsorption graph.
        atom_tags (list[int]): list of tags defining whether an atom is part of the adsorbate or the surface.
    Returns:
        (bool): True = Fragmented adsorbate
                False = Connected adsorbate
    """
    graph = extract_adsorbate(graph)
    if len(graph) == 1 and graph.number_of_edges() == 0:
        return False
    return not is_connected(graph)


def is_ring(graph: Graph) -> bool:
    """Check if the graph contains a ring."""
    adsorbate = extract_adsorbate(graph)
    cycles = list(cycle_basis(adsorbate))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    if len(ring_nodes) > 0:
        return True
    else:
        return False


def H_filter(graph: Graph) -> bool:
    """
    Graph filter that checks the connectivity of H atoms within the adsorbate 
    using a NetworkX graph object.

    Each H atom in the adsorbate must be connected to a maximum of one other 
    atom that is also part of the adsorbate.

    Args:
        graph (nx.Graph): NetworkX Graph object.
        atom_tags (Dict[int, int]): A dictionary mapping the graph's node IDs (keys)
                                    to an integer tag (values), where 1 indicates 
                                    the atom is part of the adsorbate, and 0 otherwise.

    Returns:
        bool: True = Correct connectivity for all H atoms in the adsorbate.
              False = Bad connectivity for at least one H atom in the adsorbate.
    """
    
    H_nodes_in_adsorbate = []
    for node_id in graph.nodes:
        is_hydrogen = graph.nodes[node_id].get('elem') == 'H'
        is_adsorbate = graph.nodes[node_id].get('atom_tags') == 1
        
        if is_hydrogen and is_adsorbate:
            H_nodes_in_adsorbate.append(node_id)

    for h_node in H_nodes_in_adsorbate:
        adsorbate_neighbor_count = 0
        for neighbor_node in graph.neighbors(h_node):            
            if graph.nodes[neighbor_node].get('atom_tags') == 1:
                adsorbate_neighbor_count += 1
        if adsorbate_neighbor_count > 1:
            return False
    return True

def C_filter(graph: Graph) -> bool:
    """
    Graph filter that checks the connectivity of H atoms within the adsorbate 
    using a NetworkX graph object.

    Each H atom in the adsorbate must be connected to a maximum of one other 
    atom that is also part of the adsorbate.

    Args:
        graph (nx.Graph): NetworkX Graph object.
        atom_tags (Dict[int, int]): A dictionary mapping the graph's node IDs (keys)
                                    to an integer tag (values), where 1 indicates 
                                    the atom is part of the adsorbate, and 0 otherwise.

    Returns:
        bool: True = Correct connectivity for all H atoms in the adsorbate.
              False = Bad connectivity for at least one H atom in the adsorbate.
    """
    
    H_nodes_in_adsorbate = []
    for node_id in graph.nodes:
        is_hydrogen = graph.nodes[node_id].get('elem') == 'C'
        is_adsorbate = graph.nodes[node_id].get('atom_tags') == 1
        
        if is_hydrogen and is_adsorbate:
            H_nodes_in_adsorbate.append(node_id)

    for h_node in H_nodes_in_adsorbate:
        adsorbate_neighbor_count = 0
        for neighbor_node in graph.neighbors(h_node):            
            if graph.nodes[neighbor_node].get('atom_tags') == 1:
                adsorbate_neighbor_count += 1
        if adsorbate_neighbor_count > 4:
            return False
    return True


def adsorption_filter(graph: Graph) -> bool:
    """
    Check presence of surface atoms in the adsorption graph.

    Args:
        graph(networkx.Graph): Graph of the adsorption structure obtained with atoms_to_data.
    Returns:
        (bool): True = Surface atoms present in the adsorption graph
                False = No surface atoms in the adsorption graph
    """
    return False if all([graph.nodes[node_id]["atom_tags"] == 1 for node_id in range(graph.nodes)]) else True


def ase_adsorption_filter(atoms: Atoms, atom_tags: list[int]) -> bool:
    """
    Check that the adsorbate has not been incorporated in the bulk.

    Args:
        graph (Graph): Input adsorption/molecular graph.
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


def graph_plotter(
    g: Graph,
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
        graph(networkx.Graph): Input graph obtained with atoms_to_data.
    """
    labels = get_node_attributes(g, "elem")
    node_colors = {i: RGB_COLORS[labels[i]] for i in g.nodes}
    edge_colors = ["black" for edge in g.edges]
    plt.figure(figsize=figsize, dpi=dpi)
    draw_networkx(
        g,
        labels=labels,
        node_size=node_size,
        font_color=font_color,
        font_weight=font_weight,
        node_color=list(node_colors.values()),
        edge_color=edge_colors,
        alpha=alpha,
        arrowsize=arrowsize,
        width=width,
        pos=kamada_kawai_layout(g),
        linewidths=0.5,
    )
    if node_index:
        pos_dict = kamada_kawai_layout(g)
        for node in g.nodes:
            x, y = pos_dict[node]
            plt.text(x + 0.05, y + 0.05, node, fontsize=7)
    if text != None:
        plt.text(0.03, 0.9, text, fontsize=10)
    plt.axis("off")
    plt.draw()


def connectivity_signature(g: Graph) -> list[tuple[str, tuple[str]]]:
    """
    Return a sorted list of (element, sorted neighbor elements) for each node.
    This signature is independent of atom order and can be used to assess
    if two graphs have the same connectivity configuration.
    """
    sig = []
    for n in g.nodes():
        elem = g.nodes[n]['elem']
        neighbor_elems = sorted([g.nodes[neigh]['elem'] for neigh in g.neighbors(n)])
        sig.append((elem, tuple(neighbor_elems)))
    return sorted(sig)


def get_connectivity_dict(atoms: Atoms, atom_tags: list[int], target: str="as") -> dict:
    """
    Generates a canonical signature dictionary representing the connectivity 
    between adsorbate and surface atoms.
    
    The signature is independent of atom order and can be used to assess
    if two adsorption structures have the same bonding configuration 
    (number of bonds and bonding elements).
    
    Assumes atom_tags is a list where surface atoms have one tag and adsorbate 
    atoms have a different tag (e.g., [0, 0, 0, 1, 1]).
    
    The final dictionary structure is:
    {'Element1-Element2': [[idx_A, idx_B], [idx_C, idx_D], ...]}
    where Element1-Element2 is sorted (e.g., 'C-Fe'), and the index pairs 
    [idx_A, idx_B] are sorted lists themselves, and the list of pairs is 
    sorted for canonical representation.

    Args:
        atoms (Atoms): ASE Atoms object of the adsorption structure.
        atom_tags (list[int]): List of tags defining adsorbate (1) vs surface atoms (0).
        target (str): Type of connections to consider:
                      'as' for adsorbate-surface,
                      'aa' for adsorbate-adsorbate,
                      'ss' for surface-surface.
    Returns:
        dict: Canonical connectivity signature dictionary.
    """
    graph = atoms_to_data(atoms, atom_tags=atom_tags, surface_order=1, filter=False)
    connectivity_dict = defaultdict(set)
    for node1, node2 in graph.edges():
        data1 = graph.nodes[node1]
        data2 = graph.nodes[node2]
        elem1, elem2 = data1['elem'], data2['elem']
        ase_idx1, ase_idx2 = data1['idx'], data2['idx']
        tag1, tag2 = atom_tags[node1], atom_tags[node2]
        elem_str = "-".join(sorted([elem1, elem2]))
        if target == "as" and tag1 != tag2:
            connectivity_dict[elem_str].add(tuple(sorted((ase_idx1, ase_idx2))))
        elif target == "aa" and tag1 == 1 and tag2 == 1:
            connectivity_dict[elem_str].add(tuple(sorted((ase_idx1, ase_idx2))))
        elif target == "ss" and tag1 == 0 and tag2 == 0:
            connectivity_dict[elem_str].add(tuple(sorted((ase_idx1, ase_idx2))))
        else:
            continue
    canonical_dict = {}
    for elem_str, bond_set in connectivity_dict.items():
        sorted_bonds = sorted([list(pair) for pair in bond_set])
        canonical_dict[elem_str] = sorted_bonds
        
    return canonical_dict