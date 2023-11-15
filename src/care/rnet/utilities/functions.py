from collections import namedtuple
import networkx as nx
from itertools import combinations, product
import networkx.algorithms.isomorphism as iso
from ase import Atoms
import numpy as np
from scipy.spatial import Voronoi
from rdkit import Chem
from rdkit.Chem import AllChem


CORDERO = {"Ac": 2.15, "Al": 1.21, "Am": 1.80, "Sb": 1.39, "Ar": 1.06,
           "As": 1.19, "At": 1.50, "Ba": 2.15, "Be": 0.96, "Bi": 1.48,
           "B" : 0.84, "Br": 1.20, "Cd": 1.44, "Ca": 1.76, "C" : 0.76,
           "Ce": 2.04, "Cs": 2.44, "Cl": 1.02, "Cr": 1.39, "Co": 1.50,
           "Cu": 1.32, "Cm": 1.69, "Dy": 1.92, "Er": 1.89, "Eu": 1.98,
           "F" : 0.57, "Fr": 2.60, "Gd": 1.96, "Ga": 1.22, "Ge": 1.20,
           "Au": 1.36, "Hf": 1.75, "He": 0.28, "Ho": 1.92, "H" : 0.31,
           "In": 1.42, "I" : 1.39, "Ir": 1.41, "Fe": 1.52, "Kr": 1.16,
           "La": 2.07, "Pb": 1.46, "Li": 1.28, "Lu": 1.87, "Mg": 1.41,
           "Mn": 1.61, "Hg": 1.32, "Mo": 1.54, "Ne": 0.58, "Np": 1.90,
           "Ni": 1.24, "Nb": 1.64, "N" : 0.71, "Os": 1.44, "O" : 0.66,
           "Pd": 1.39, "P" : 1.07, "Pt": 1.36, "Pu": 1.87, "Po": 1.40,
           "K" : 2.03, "Pr": 2.03, "Pm": 1.99, "Pa": 2.00, "Ra": 2.21,
           "Rn": 1.50, "Re": 1.51, "Rh": 1.42, "Rb": 2.20, "Ru": 1.46,
           "Sm": 1.98, "Sc": 1.70, "Se": 1.20, "Si": 1.11, "Ag": 1.45,
           "Na": 1.66, "Sr": 1.95, "S" : 1.05, "Ta": 1.70, "Tc": 1.47,
           "Te": 1.38, "Tb": 1.94, "Tl": 1.45, "Th": 2.06, "Tm": 1.90,
           "Sn": 1.39, "Ti": 1.60, "Wf": 1.62, "U" : 1.96, "V" : 1.53,
           "Xe": 1.40, "Yb": 1.87, "Y" : 1.90, "Zn": 1.22, "Zr": 1.75}


MolPack = namedtuple('MolPack',['code', 'ase_mol','nx_graph', 'num_H_removed'])

def rdkit_to_ase(rdkit_molecule) -> Atoms:
    """
    Generate an ASE Atoms object from an RDKit molecule.
    """
    AllChem.EmbedMolecule(rdkit_molecule, AllChem.ETKDG())
    positions = rdkit_molecule.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in rdkit_molecule.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)

def ase_2_graph(atoms: Atoms, coords: bool=False) -> nx.Graph:
    """
    Generate a NetworkX Graph (undirected) from an ASE Atoms object.

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

def generate_pack(rdkit_molecule: Chem,
                  group: int) -> dict[int, list[MolPack]]:
    """
    Given a molecule, generate all the possible intermediates by removing up to n_H hydrogens.

    Parameters
    ----------
    molecule : Chem
        RDKit object of the molecule.
    group : int
        Group number of the molecule. For labeling purposes
    """
    rdkit_molecule = Chem.AddHs(rdkit_molecule)
    Chem.SanitizeMol(rdkit_molecule)
    
    # Convert to ASE object
    molecule = rdkit_to_ase(rdkit_molecule)
    nH = molecule.get_chemical_symbols().count("H")
    if molecule.get_chemical_formula() == 'H2':
        nH = 1
    molecule.arrays["conn_pairs"] = get_voronoi_neighbourlist(molecule, 0.25, 1.0, ['C', 'H', 'O']) 
    # note: MolPack = namedtuple('MolPack',['code', 'ase_mol','nx_graph', 'num_H_removed'])
    mg_pack = {0: [MolPack(code_name(molecule, group, 1), molecule, ase_2_graph(molecule), {})]}

    for index in range(nH):        
        pack = get_all_subs(rdkit_molecule, index + 1, 'H')
        unique = get_unique(pack[1], 'H')
        
        tmp_pack = []
        for i_code, struct in enumerate(unique):
            mol_sel = pack[0][struct]
            tmp_pack.append(MolPack(code_name(mol_sel, group, i_code+1), mol_sel, pack[1][struct], {'H': index+1}))

        mg_pack[index+1] = tmp_pack
    return mg_pack


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
    return CORDERO[element_i] + CORDERO[element_j] + tolerance


def code_name(molecule: Atoms, group: int, index: int) -> str:
    characters = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]
    n = len(characters)
    
    nC = molecule.get_chemical_symbols().count("C")
    nH = molecule.get_chemical_symbols().count("H")
    nO = molecule.get_chemical_symbols().count("O")

    if nC >= n*n or nH >= n*n or nO >= n*n or group >= n*n or index >= n*n:
        return "Atom count exceeds encoding capacity"

    def double_encode(x):
        return characters[x // n] + characters[x % n]
    
    encoded_name = double_encode(nC) + double_encode(nH) + double_encode(nO) + double_encode(group) + double_encode(index)

    if len(encoded_name) != 10:
        return "Invalid encoded string length"
    
    return f"{encoded_name}"


def search_code(mg_pack: dict[int, list[MolPack]], code: str):
    for _, pack in mg_pack.items():
         for obj in pack:
            if str(obj.code) == str(code):
                return obj
    else:
        return False
    

def draw_dot(graph, filename):
    p=nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(filename)

def generate_range(ase_molecule: Atoms, 
                   element: str, 
                   subs: int, 
                   group: int) -> dict[int, list]: #generating all the possibilities

    mg_pack = {0: [MolPack(code_name(ase_molecule, group, 1), ase_molecule, ase_2_graph(ase_molecule, coords=False), {})]}

    for index in range(subs):
        
        pack = get_all_subs(ase_molecule, index + 1, element)
        unique = get_unique(pack[1], element)
        
        tmp_pack = []
        for i_code, struct in enumerate(unique):
            mol_sel = pack[0][struct]
            tmp_pack.append(MolPack(code_name(mol_sel, group, i_code+1), mol_sel, pack[1][struct], {'H': index+1}))

        mg_pack[index+1] = tmp_pack

    return mg_pack

def get_all_subs(rdkit_molecule: Chem, 
                 n_sub: int, 
                 element: str) -> tuple[list[Atoms], list[nx.Graph]]:
    """
    function to what point you want to remove hydrogens
    """
    
    elem_indices = [atom.GetIdx() for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == element]
    mol_pack, graph_pack = [], []

    for comb in combinations(elem_indices, n_sub):

        new_mol = Chem.RWMol(rdkit_molecule)
    
        # Sort and reverse to ensure we remove atoms without affecting the indices of atoms yet to be removed
        for index in reversed(sorted(comb)):
            atom = new_mol.GetAtomWithIdx(index)
            
            # Skip removal if the atom has no neighbors
            if atom.GetDegree() == 0:
                print(f"WARNING: not removing {atom.GetSymbol()} atom without neighbors")
                continue
            new_mol.RemoveAtom(index)

        # Sanitize the molecule
        Chem.SanitizeMol(new_mol)
        # Skip removal if the atom has no neighbors
        
        # Convert the sanitized RDKit molecule back to ASE for further tasks
        new_ase_mol = rdkit_to_ase(new_mol)
        new_ase_mol.arrays["conn_pairs"] = get_voronoi_neighbourlist(new_ase_mol, 0.25, 1.0, ['C', 'H', 'O'])
        
        mol_pack.append(new_ase_mol)
        graph_pack.append(ase_2_graph(new_ase_mol, coords=False))
    return mol_pack, graph_pack


def get_unique(graph_pack: list[nx.Graph], element: str) -> list[int]:
    """
    Function to get unique graph configurations based on isomorphism checks.
    """
    accepted = []
    em = iso.categorical_node_match('elem', element)
    for index, graph1 in enumerate(graph_pack):
        if all(not nx.is_isomorphic(graph1, graph_pack[accepted_idx], node_match=em) for accepted_idx in accepted):
            accepted.append(index)
    return accepted


