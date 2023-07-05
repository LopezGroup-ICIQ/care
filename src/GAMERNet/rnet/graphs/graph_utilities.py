from itertools import product
from scipy.spatial import Voronoi
import networkx as nx
import ase
import numpy as np
import json as js

# Loading parameter file
with open('data/parameters.json') as f:
    params = js.load(f)

# Atomic radii from Cordero
CORDERO = params['CORDERO']

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