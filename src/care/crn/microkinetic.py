"""Helper functions for running microkinetic simulations."""

import numpy as np
from rdkit import Chem

def stoic_forward(matrix: np.ndarray) -> np.ndarray:
    """
    Filter function for the stoichiometric matrix.
    Negative elements are considered and changed of sign in order to
    compute the direct reaction rates.
    Args:
        matrix(ndarray): Stoichiometric matrix
    Returns:
        mat(ndarray): Filtered matrix for constructing forward reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] < 0:
                mat[i][j] = -matrix[i][j]
    return mat


def stoic_backward(matrix: np.ndarray) -> np.ndarray:
    """
    Filter function for the stoichiometric matrix.
    Positive elements are considered and kept in order to compute
    the reverse reaction rates.
    Args:
        matrix(ndarray): stoichiometric matrix
    Returns:
        mat(ndarray): Filtered matrix for constructing reverse reaction rates.
    """
    mat = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if matrix[i][j] > 0:
                mat[i][j] = matrix[i][j]
    return mat

def net_rate(y: np.ndarray, 
             kd: np.ndarray, 
             ki: np.ndarray, 
             v_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the net reaction rate for each elementary reaction.
    Args:
        y(ndarray): surface coverage + partial pressures array [-/Pa].
        kd, kr(ndarray): kinetic constants of the direct/reverse steps.
        v_matrix(ndarray): stoichiometric matrix of the system.
    Returns:
        (ndarray): Net reaction rate of the elementary reactions [1/s].
    """
    v_ff = stoic_forward(v_matrix)
    v_bb = stoic_backward(v_matrix)
    return kd * np.prod(y**v_ff.T, axis=1) - ki * np.prod(y**v_bb.T, axis=1)

from rdkit import Chem

def iupac_to_inchikey(iupac_name: str) -> str:
    mol = Chem.MolFromIUPACName(iupac_name)
    if mol is not None:
        return Chem.inchi.MolToInchiKey(mol)
    else:
        return "Invalid IUPAC name"

