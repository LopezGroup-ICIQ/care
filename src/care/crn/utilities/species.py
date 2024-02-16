from itertools import product

import networkx as nx
import numpy as np
from ase import Atom, Atoms
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import Voronoi

from care.constants import CORDERO

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def edge_cutoffs(
    node_i: nx.Graph.nodes, node_j: nx.Graph.nodes, tolerance: float
) -> float:
    """
    Get the cutoff distance for two atoms to be considered connected using Cordero's atomic radii.

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
    return CORDERO[element_i] + CORDERO[element_j] + tolerance


def get_voronoi_neighbourlist(
    atoms: Atoms, tolerance: float, scaling_factor: float, molecule_elements: list[str]
) -> np.ndarray:
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
    # corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(
        pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0
    )
    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their covalent radii plus a tolerance.
    # NB surface atoms covalent radii are scaled by a factor corresponding to scaling_factor.
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = (
            CORDERO[atoms[pair[0]].symbol]
            + CORDERO[atoms[pair[1]].symbol]
            + tolerance
            + (scaling_factor - 1.0)
            * (
                (atoms[pair[0]].symbol not in molecule_elements)
                * CORDERO[atoms[pair[0]].symbol]
                + (atoms[pair[1]].symbol not in molecule_elements)
                * CORDERO[atoms[pair[1]].symbol]
            )
        )
        if distance <= threshold:
            pairs_lst.append(pair)
    return np.sort(np.array(pairs_lst), axis=1)


def atoms_to_graph(atoms: Atoms, coords: bool = False) -> nx.Graph:
    """
    Generates a NetworkX Graph from an ASE Atoms object.

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
            num: {"elem": elems_list[i], "xyz": xyz_coords[i]}
            for i, num in enumerate(num_atom)
        }
    else:
        node_attrs = {num: {"elem": elems_list[i]} for i, num in enumerate(num_atom)}
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


def rdkit_to_graph(mol: Chem.Mol) -> nx.Graph:
    """
    Generates a NetworkX Graph from an RDKit molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    nx.Graph
        NetworkX Graph of the molecule (with atomic coordinates and bond lengths).
    """

    # Adding Hs to the molecule
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    nx_graph = nx.Graph()

    for i, atom in enumerate(mol.GetAtoms()):
        nx_graph.add_node(atom.GetIdx(), elem=atom.GetSymbol(), xyz=positions[i])

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = positions[i].Distance(positions[j])
        nx_graph.add_edge(i, j, length=length)

    return nx_graph


def generate_alkanes_recursive(n_carbon: int, main_chain="") -> list[str]:
    """
    Generate all possible linear and branched alkanes with a given number of carbon atoms.

    Parameters
    ----------
    n_carbon : int
        Maximum number of carbon atoms in the alkanes.
    main_chain : str, optional
        String where SMILES will be generated, by default ""

    Returns
    -------
    unique_alkanes : list[str]
        List of unique alkanes in SMILES format.
    """
    if n_carbon == 0:
        return [main_chain]
    if n_carbon < 0:
        return []

    alkanes = []

    # Continue the main chain
    new_chain = main_chain + "C"
    alkanes += generate_alkanes_recursive(n_carbon - 1, new_chain)

    # Add branches
    if len(main_chain) > 0:
        for i in range(len(main_chain)):
            if main_chain[i] == "C":
                new_chain = main_chain[:i] + "C(C)" + main_chain[i + 1 :]
                alkanes += generate_alkanes_recursive(n_carbon - 1, new_chain)

    # Remove duplicates
    unique_alkanes = list(set(alkanes))
    return unique_alkanes


def canonicalize_smiles(smiles_list: list[str], rmv_quatr_C: bool) -> list[str]:
    """
    Canonicalize a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    list[str]
        List of unique SMILES strings.
    """
    canonical_smiles_set = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # If the molecule contains a quaternary carbon, do not add it to the list
            if rmv_quatr_C:
                if any(atom.GetDegree() == 4 for atom in mol.GetAtoms()):
                    continue
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            canonical_smiles_set.add(canonical_smiles)
    return list(canonical_smiles_set)


def gen_alkanes(nc: int) -> tuple[list[str], list[Chem.Mol]]:
    """
    Generate all possible alkanes from 1 Carbon atom to the given number of carbon atoms.

    Parameters
    ----------
    nc : int
        Maximum number of carbon atoms in the alkanes.

    Returns
    -------
    unique_alkanes : list[str]
        List of unique alkanes in SMILES format.
    mol_alkanes : list[Chem.Mol]
        List of RDKit molecules of alkanes.
    """
    all_alkanes = []
    for i in range(1, nc + 1):
        all_alkanes += generate_alkanes_recursive(i)
    unique_alkanes = canonicalize_smiles(all_alkanes, rmv_quatr_C=False)
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in unique_alkanes]
    return unique_alkanes, mol_alkanes


def gen_epoxides(mol_alkanes: list, no: int) -> list[str]:
    """
    Generate all possible epoxides smiles based on the given number of carbon and oxygen atoms.

    Parameters
    ----------
    mol_alkanes : list
        List of RDKit molecules of alkanes.
    no : int
        Maximum number of oxygen atoms in the epoxides.

    Returns
    -------
    list[str]
        List of epoxides SMILES strings.
    """

    epoxides = []
    cond1 = lambda atoms: atoms[0].GetDegree() < 4 and atoms[1].GetDegree() < 4
    if no == 0:
        return epoxides
    for mol in mol_alkanes:
        # generate list of tuple of adjacent atoms satisfying the cond1
        c_pairs = [
            (atom, nbr)
            for atom in mol.GetAtoms()
            for nbr in atom.GetNeighbors()
            if cond1((atom, nbr))
        ]
        for c_pair in c_pairs:
            # generate epoxide
            tmp_mol = Chem.RWMol(mol)
            oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
            tmp_mol.AddBond(
                c_pair[0].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.AddBond(
                c_pair[1].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.UpdatePropertyCache()
            smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
            epoxides.append(smiles)
    return list(set(epoxides))


def gen_ethers(mol_alkanes: list, noc: int) -> tuple[list[str], list[Chem.Mol]]:
    """
    Add an oxygen atom to an alkane to generate an ether.

    Parameters
    ----------
    alkane_smiles_list : list[str]
        List of alkane SMILES strings.

    Returns
    -------
    unique_ethers : list[str]
        List of ether SMILES strings.
    mol_ethers : list[Chem.Mol]
        List of RDKit molecules of ethers.
    """

    ethers = []
    if noc == 0:
        return ethers, []

    for mol in mol_alkanes:
        # generate list of tuple of adjacent atoms satisfying the cond1
        c_pairs = [
            (atom, nbr) for atom in mol.GetAtoms() for nbr in atom.GetNeighbors()
        ]
        for c_pair in c_pairs:
            # Insert oxygen atom between the two carbon atoms
            tmp_mol = Chem.RWMol(mol)
            oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
            tmp_mol.AddBond(
                c_pair[0].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.AddBond(
                c_pair[1].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            # Delete the bond between the two carbon atoms
            tmp_mol.RemoveBond(c_pair[0].GetIdx(), c_pair[1].GetIdx())
            tmp_mol.UpdatePropertyCache()
            smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
            ethers.append(smiles)
    unique_ethers = list(set(ethers))
    mol_ethers = [Chem.MolFromSmiles(smiles) for smiles in unique_ethers]
    return unique_ethers, mol_ethers


def add_oxygens(mol: Chem.Mol, noc: int) -> set[str]:
    """
    Add up to 'noc' oxygen atoms to suitable carbon atoms in the molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    noc : int
        Maximum number of oxygens to add. If noc < 0, add as many oxygens as possible.

    Returns
    -------
    unique_molecules : set[str]
        Set of all possible molecules with added oxygens (SMILES format).
    """
    unique_molecules = set()

    # Filter out carbon atoms with no free sites
    suitable_carbons = [
        (atom.GetIdx(), 4 - atom.GetDegree())
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetDegree() < 4
    ]
    if not suitable_carbons:
        return unique_molecules

    # Generate combinations
    combos = [list(range(num + 1)) for _, num in suitable_carbons]

    for combo in product(*combos):
        total_oxygens = sum(combo)
        if total_oxygens == 0 or (noc >= 0 and total_oxygens > noc):
            continue

        tmp_mol = Chem.RWMol(mol)
        for num_oxygens, (carbon_idx, _) in zip(combo, suitable_carbons):
            for _ in range(num_oxygens):
                oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
                tmp_mol.AddBond(
                    carbon_idx, oxygen_idx, order=Chem.rdchem.BondType.SINGLE
                )

        tmp_mol.UpdatePropertyCache()
        smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
        unique_molecules.add(smiles)

    return unique_molecules


def get_fragment_energy(atoms: Atoms) -> float:
    """
    Calculate fragment energy from closed shell structures.
    This function allows to calculate the energy of both open- and closed-shell structures,
    keeping the same reference.

    Parameters:
    ----------
        atoms (ase.Atoms): Atoms object of the structure to evaluate

    Returns:
    -------
        float: Energy of the structure.
    """
    # Count elemens in the structure
    n_C = atoms.get_chemical_symbols().count("C")
    n_H = atoms.get_chemical_symbols().count("H")
    n_O = atoms.get_chemical_symbols().count("O")
    n_N = atoms.get_chemical_symbols().count("N")
    n_S = atoms.get_chemical_symbols().count("S")

    # Reference DFT energy for C, H, O, N, S
    e_H2O = -14.21877278  # O
    e_H2 = -6.76639487  # H
    e_NH3 = -19.54236910  # N
    e_H2S = -11.20113092  # S
    e_CO2 = -22.96215586  # C
    return (
        n_C * e_CO2
        + (n_O - 2 * n_C) * e_H2O
        + (4 * n_C + n_H - 2 * n_O - 3 * n_N - 2 * n_S) * e_H2 * 0.5
        + (n_N * e_NH3)
        + (n_S * e_H2S)
    )
