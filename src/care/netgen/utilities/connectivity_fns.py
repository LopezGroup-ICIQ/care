from ase import Atoms
import numpy as np
from itertools import product
from scipy.spatial import Voronoi

from care.netgen.data.constants_and_data import CORDERO

def get_voronoi_neighbourlist(atoms: Atoms,
                              tolerance: float,
                              scaling_factor: float,
                              molecule_elements: list[str]) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The catalyst surface does not contain elements present in the adsorbate.
    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.
    """

    if len(atoms) == 1:
        return np.array([])
    # First condition for two atoms to be connected: They must share a Voronoi facet
    coords_arr = np.repeat(np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    # corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)
    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their covalent radii plus a tolerance.
    # NB surface atoms covalent radii are scaled by a factor corresponding to scaling_factor.
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = CORDERO[atoms[pair[0]].symbol] + CORDERO[atoms[pair[1]].symbol] + tolerance + \
                    (scaling_factor - 1.0) * ((atoms[pair[0]].symbol not in molecule_elements) * CORDERO[atoms[pair[0]].symbol] + \
                                              (atoms[pair[1]].symbol not in molecule_elements) * CORDERO[atoms[pair[1]].symbol])
        if distance <= threshold:
            pairs_lst.append(pair)
    return np.sort(np.array(pairs_lst), axis=1)