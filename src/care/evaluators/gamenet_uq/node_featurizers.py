"""
Node featuruzation for PyG
"""

import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data

from care.gnn.graph import get_voronoi_neighbourlist


def get_magnetization(graph: Data):
    if graph.metal in ("Fe", "Co", "Ni"):
        graph.x = torch.cat((graph.x, torch.ones((graph.x.shape[0], 1))), dim=1)
    else:
        graph.x = torch.cat((graph.x, torch.zeros((graph.x.shape[0], 1))), dim=1)
    return graph


def adsorbate_node_featurizer(graph: Data, adsorbate_elements: list[str]) -> Data:
    """
    Add a node feature to the graph which is 1 if the node is an adsorbate atom, 0 otherwise.
    graph must have as attributes:
    - atoms_obj (Atoms): ASE atoms object containing a slab with an adsorbate
    - node_feats (list[str]): list of node features to be used in the graph
    """
    x_adsorbate = torch.zeros((graph.x.shape[0], 1))
    for i, node in enumerate(graph.x):
        index = torch.where(node == 1)[0][0].item()
        x_adsorbate[i, 0] = 1 if graph.node_feats[index] in adsorbate_elements else 0
    graph.x = torch.cat((graph.x, x_adsorbate), dim=1)
    return graph


def get_gcn(
    graph: Data,
    atoms: Atoms,
    adsorbate_elements: list[str],
    surface_neighbours: list[int],
) -> Data:
    """
    Return the (normalized) generalized coordination number (gcn) for each surface atom in the ASE Atoms object.
    gcn is defined as the sum of the coordination numbers (cn) of the neighbours divided by the maximum coordination number.
    gcn=0 atom alone; gcn=1 bulk atom; 0<gcn<1=surface atom.

    graph must have as attributes:
    - atoms (Atoms): ASE atoms object containing a slab with an adsorbate
    - node_feats (list[str]): list of node features to be used in the graph

    Args:
        atoms (Atoms): ASE atoms object containing a slab with an adsorbate
        adsorbate_elements (list[str]): list of symbols of the adsorbate elements

    Returns:
        Data: PyG Data object with the gcn as a node feature. Data.x.shape[1] increases by 1.
                Data.node_feats is also updated.
    """
    if all(graph.elem[i] in adsorbate_elements for i in range(len(graph.elem))):
        graph.x = torch.cat((graph.x, torch.zeros((graph.x.shape[0], 1))), dim=1)
        graph.node_feats.append("gcn")
        return graph
    y = get_voronoi_neighbourlist(
        atoms, 0.5, 1.0, adsorbate_elements
    )  # only slab atoms are considered
    adsorbate_elements_indices = [
        graph.node_feats.index(element) for element in adsorbate_elements
    ]
    neighbour_dict = {}
    for idx, atom in enumerate(atoms):
        cn = 0
        neighbour_list = []
        for row in y:
            if idx in row:
                neighbour_index = row[0] if row[0] != idx else row[1]
                if atoms[neighbour_index].symbol not in adsorbate_elements:
                    cn += 1
                    neighbour_list.append(
                        (
                            atoms[neighbour_index].symbol,
                            neighbour_index,
                            atoms[neighbour_index].position[2],
                        )
                    )
            else:
                continue
        neighbour_dict[idx] = (cn, atom.symbol, neighbour_list)
    max_cn = max([neighbour_dict[i][0] for i in neighbour_dict.keys()])
    gcn_dict = {}
    for idx in neighbour_dict.keys():
        if atoms[idx].symbol in adsorbate_elements:
            gcn_dict[idx] = (None, neighbour_dict[idx][0])
            continue
        cn_sum = 0.0
        for neighbour in neighbour_dict[idx][2]:
            cn_sum += neighbour_dict[neighbour[1]][0]
        gcn_dict[idx] = (cn_sum / max_cn**2, neighbour_dict[idx][0])
    gcn = torch.zeros((graph.x.shape[0], 1))
    counter = 0
    for i, node in enumerate(graph.x):
        index = torch.where(node == 1)[0][0].item()
        if index not in adsorbate_elements_indices:
            gcn[i] = gcn_dict[surface_neighbours[counter]][0]
            counter += 1
    graph.x = torch.cat((graph.x, gcn), dim=1)
    graph.node_feats.append("gcn")
    return graph


def get_radical_atoms(atoms_obj: Atoms, adsorbate_elements: list[str]) -> list[int]:
    """
    Detect atoms in the molecule which are radicals with RDKit.

    Args:
        atoms_obj (ase.Atoms): ASE atoms object of the adsorption structure
        adsorbate_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])

    Returns:
        radical_atoms (list[int]): List of indices of the radical atoms in the atoms_obj.
    """

    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = "\n".join(
        f"{symbol} {x} {y} {z}"
        for symbol, (x, y, z) in zip(atomic_symbols, coordinates)
    )
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(
        conn_mol, Chem.SANITIZE_FINDRADICALS ^ Chem.SANITIZE_SETHYBRIDIZATION
    )
    radical_atoms = [
        atom.GetIdx()
        for atom in conn_mol.GetAtoms()
        if atom.GetNumRadicalElectrons() > 0
    ]
    return radical_atoms


def get_atom_valence(atoms_obj: Atoms, adsorbate_elements: list[str]) -> list[float]:
    """
    For each atom in the adsorbate, calculate the valence.
    Valence is defined as (x_max - x) / x_max, where x is the degree of the atom,
    and x_max is the maximum degree of the atom in the molecule. Bond order is not taken into account.
    valence=0 atom alone; valence=1 fully saturated atom.

    Ref: https://doi.org/10.1103/PhysRevLett.99.016105

    Args:
        atoms_obj (ase.Atoms): ASE atoms object.
        molecule_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])
    Returns:
        valence (list[float]): List of valences for each atom in the molecule.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = "\n".join(
        f"{symbol} {x} {y} {z}"
        for symbol, (x, y, z) in zip(atomic_symbols, coordinates)
    )
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(
        conn_mol, Chem.SANITIZE_FINDRADICALS ^ Chem.SANITIZE_SETHYBRIDIZATION
    )
    degree_vector = np.vectorize(lambda x: x.GetDegree())
    max_degree_vector = np.vectorize(
        lambda x: Chem.GetPeriodicTable().GetDefaultValence(x.GetAtomicNum())
    )
    atom_array = np.array([i for i in conn_mol.GetAtoms()]).reshape(-1, 1)
    degree_array = degree_vector(atom_array) / max_degree_vector(
        atom_array
    )  # valence = x / x_max in order to have the same trend as gcn
    valence = degree_array.reshape(-1, 1)
    return valence
