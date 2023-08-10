from itertools import product
from scipy.spatial import Voronoi
import networkx as nx
import ase
import numpy as np
import json as js

# Atomic radii from Cordero
CORDERO = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44, 'O' : 0.66,
           'Pd': 1.39, 'P' : 1.07, 'Pt': 1.36, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42, 'Rb': 2.20, 'Ru': 1.46,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22, 'Zr': 1.75}  # Atomic radii from Cordero 

def edge_cutoffs(node_i: nx.Graph.nodes, node_j: nx.Graph.nodes, tolerance: float) -> float:
    """Get the cutoff distance for two atoms to be considered connected using Cordero's atomic radii.

    Parameters
    ----------
    node_i : nx.Graph.nodes
        Node i.
    node_j : nx.Graph.nodes
        Node j.
    tolerance : float
        Tolerance for the cutoff distance.

    Returns
    -------
    float
        Cutoff distance.
    """

    element_i = node_i.symbol
    element_j = node_j.symbol
    cutoff = (CORDERO[element_i] + CORDERO[element_j]) + tolerance
    return cutoff

# Modified version of the original function provided by Santiago Morandi
def get_voronoi_neighbourlist(atoms: ase.Atoms, 
                              tolerance: float, 
                              ) -> np.ndarray:
    """Get the connectivity list from a Voronoi analysis, considering periodic boundary conditions (PBC).
    To have two atoms connected, these must satisfy two conditions:
    1. They must share a Voronoi facet.
    2. The distance between them must be less than the sum of their covalent radii (plus a tolerance).

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object of the system.
    tolerance : float
        Tolerance for condition 2.

    Returns
    -------
    np.ndarray
        N_edges x 2 array with the connectivity list.
        Note: The array contains all the edges only in one direction.
    """
    
    # First condition to have two atoms connected: They must share a Voronoi facet
    coords_arr = np.copy(atoms.get_scaled_positions())  
    coords_arr = np.expand_dims(coords_arr, axis=0)
    coords_arr = np.repeat(coords_arr, 27, axis=0)
    mirrors = [-1, 0, 1]
    mirrors = np.asarray(list(product(mirrors, repeat=3)))
    mirrors = np.expand_dims(mirrors, 1)
    mirrors = np.repeat(mirrors, coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors,
                                  (coords_arr.shape[0] * coords_arr.shape[1],
                                   coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    true_arr = pairs_corr[:, 0] == pairs_corr[:, 1]
    true_arr = np.argwhere(true_arr)
    pairs_corr = np.delete(pairs_corr, true_arr, axis=0)
    
    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their
    # covalent radii plus a tolerance.
    dst_d = {}
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)  # mic=True for PBC
        elem_pair = (atoms[pair[0]].symbol, atoms[pair[1]].symbol)
        fr_elements = frozenset(elem_pair)
        if elem_pair not in dst_d:
            dst_d[fr_elements] = CORDERO[atoms[pair[0]].symbol] + CORDERO[atoms[pair[1]].symbol] + tolerance
        if distance <= dst_d[fr_elements]:
            pairs_lst.append(pair)
    if len(pairs_lst) == 0:
        return np.array([])
    else:
        return np.sort(np.array(pairs_lst), axis=1)

def ase_2_graph(atoms: ase.Atoms, coords: bool) -> nx.Graph:
    """Generates a NetworkX Graph from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object of the molecule.
    coords : bool
        Boolean indicating whether to include the atomic coordinates in the graph.

    Returns
    -------
    nx.Graph
        NetworkX Graph of the molecule (with atomic coordinates and bond lengths if 'coords' is True).
    """

    num_atom = list(range(len(atoms)))
    elems_list = atoms.get_chemical_symbols()
    xyz_coords = atoms.get_positions()

    # Generating the graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(num_atom)

    if coords:
        node_attrs = {
            num: {'elem': elems_list[i], 'xyz': xyz_coords[i]}
                  for i, num in enumerate(num_atom)
                  }
    else:
        node_attrs = {
            num: {'elem': elems_list[i]}
            for i, num in enumerate(num_atom)
        }
    nx.set_node_attributes(nx_graph, node_attrs)

    # Adding the edges
    edge_attrs = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)): 
            cutoff = edge_cutoffs(atoms[i], atoms[j], tolerance=0.2)
            bond_length = atoms.get_distance(i, j)
            if bond_length < cutoff:
                edge_attrs[(i, j)] = {"length": bond_length}
    
    edges = list(edge_attrs.keys())
    nx_graph.add_edges_from(edges)
    nx.set_edge_attributes(nx_graph, edge_attrs)

    return nx_graph