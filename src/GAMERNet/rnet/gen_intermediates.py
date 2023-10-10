from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms, Atom
from itertools import product, combinations
import networkx as nx
import copy

from GAMERNet.rnet.utilities.functions import generate_map, generate_pack, MolPack

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

def generate_alkanes_recursive(G: nx.Graph, remaining_c: int, unique_alkanes: list[str]):
    """
    Generate all alkanes with a given number of carbon atoms in smiles format.
    """
    if remaining_c == 0:
        if nx.is_connected(G) and nx.is_tree(G):
            mol = nx_to_mol(G)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            unique_alkanes.add(smiles)
        return
    
    for node in G.nodes():
        if G.degree(node) < 4:  
            for neighbor in range(max(G.nodes()) + 1, max(G.nodes()) + 1 + remaining_c):
                G_new = copy.deepcopy(G)
                G_new.add_edge(node, neighbor)
                generate_alkanes_recursive(G_new, remaining_c - 1, unique_alkanes)

def nx_to_mol(G):
    mol = Chem.RWMol()
    node_to_idx = {}
    
    for node in G.nodes():
        idx = mol.AddAtom(Chem.Atom("C"))
        node_to_idx[node] = idx
    
    for edge in G.edges():
        mol.AddBond(node_to_idx[edge[0]], node_to_idx[edge[1]], Chem.BondType.SINGLE)
    
    Chem.SanitizeMol(mol)
    return mol

def rdkit_to_ase(rdkit_molecule):
    # Generate 3D coordinates for the molecule
    rdkit_molecule = Chem.AddHs(rdkit_molecule)  # Add hydrogens if not already added
    AllChem.EmbedMolecule(rdkit_molecule, AllChem.ETKDG())

    # Get the number of atoms in the molecule
    num_atoms = rdkit_molecule.GetNumAtoms()

    # Initialize lists to store positions and symbols
    positions = []
    symbols = []

    # Extract atomic positions and symbols
    for atom_idx in range(num_atoms):
        atom_position = rdkit_molecule.GetConformer().GetAtomPosition(atom_idx)
        atom_symbol = rdkit_molecule.GetAtomWithIdx(atom_idx).GetSymbol()
        positions.append(atom_position)
        symbols.append(atom_symbol)

    # Create an ASE Atoms object
    ase_atoms = Atoms([Atom(symbol=symbol, position=position) for symbol, position in zip(symbols, positions)])

    return ase_atoms

def add_oxygens_to_molecule(mol: Chem.Mol) -> set[str]:
    """Add as many oxygen atoms as possible to suitable carbon atoms in the molecule."""
    unique_molecules = set()

    # All RDkit molecules are unsaturated, i.e., C2H6 is represented as C2 (no H atoms)
    suitable_carbons = [(atom.GetIdx(), 4 - atom.GetDegree()) for atom in mol.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]

    # Generate all combinations of adding 0 to 'num_free_sites' oxygens for each suitable carbon
    combos = [list(range(num+1)) for _, num in suitable_carbons]

    for combo in product(*combos):
        total_oxygens = sum(combo)
        if total_oxygens == 0:  # Skip molecules with no oxygens added
            continue
            
        tmp_mol = Chem.RWMol(mol) # readible and writable mol object
        for (num_oxygens, (carbon_idx, _)) in zip(combo, suitable_carbons):
            for _ in range(num_oxygens):
                oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
                tmp_mol.AddBond(carbon_idx, oxygen_idx, order=Chem.rdchem.BondType.SINGLE)

        # Update explicit and implicit valence info
        tmp_mol.UpdatePropertyCache()

        # Convert to canonical SMILES for uniqueness
        smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
        unique_molecules.add(smiles)

    return unique_molecules


def id_group_dict(molecules_dict: dict[str, dict]) -> dict[str, list[str]]:
    """Corrects the labeling for isomeric systems.

    Parameters
    ----------
    molecule_dict : dict
        Dictionary containing the molecular formulas and graph for the all the case study systems.
    
    Returns
    -------
    dict
        Dictionary containing the isomeric groups.
    
    Examples
    --------
    1-propanol -> 381101
    2-propanol -> 381201
    
    Note: each label consists of 6 digits. The first three define the number of C, H, and O atoms, respectively.
    the fourth digit is the isomer tag, and the last two digits are TODO: what are the last two digits?
    """
    molec_dict = {}
    molec_combinations = combinations(molecules_dict.keys(), 2)

    for molec_1, molec_2 in molec_combinations:
        if molecules_dict[molec_1]['Formula'] != molecules_dict[molec_2]['Formula']:
            continue
        graph_1 = molecules_dict[molec_1]['Graph']
        graph_2 = molecules_dict[molec_2]['Graph']
        if not nx.is_isomorphic(graph_1, graph_2, node_match=lambda x, y: x["elem"] == y["elem"]):
            molec_dict.setdefault(molec_1, []).append(molec_1)
            molec_dict.setdefault(molec_1, []).append(molec_2)
            molec_dict.setdefault(molec_2, []).append(molec_1)
            molec_dict.setdefault(molec_2, []).append(molec_2)

    for molec in molecules_dict.keys():
        if molec not in molec_dict:
            molec_dict[molec] = [molec]

    return molec_dict

def atoms_2_graph(atoms: Atoms, coords: bool) -> nx.Graph:
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

def gen_alkanes_smiles(n_carbon: int) -> set[str]:
    """Generate all alkanes with a given number of carbon atoms in smiles format."""
    alkanes_smiles = set()
    for nC in range(n_carbon):  # Generating alkanes with 1 to n_carbon atoms
        G = nx.Graph()
        G.add_node(0)  # Start with a single carbon atom
        generate_alkanes_recursive(G, nC, alkanes_smiles)
    return alkanes_smiles

def generate_intermediates(n_carbon: int) -> tuple[dict[str, dict[int, list[MolPack]]], dict[str, dict[int, nx.DiGraph]]]:
    """
    Generates all the possible intermediates for a given number of carbon atoms 
    starting from the set of fully saturated CHO molecules (alkanes, alcohols, etc.).

    Parameters
    ----------
    n_carbon : int
        Maximum number of carbon atoms in the intermediates.

    Returns
    -------
    tuple[dict, dict]
        
    """

    # 1) Generate all closed-shell satuarated CHO molecules (alkanes and alcohols plus H2, H2O2 and O2)
    alkanes_smiles = gen_alkanes_smiles(n_carbon)    
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in list(alkanes_smiles)]
    alcohols_smiles = [add_oxygens_to_molecule(mol) for mol in mol_alkanes] 
    alcohols_smiles = [smiles for smiles_set in alcohols_smiles for smiles in smiles_set]  # flatten list of lists
    alcohols_smiles += ['O', 'OO', '[H][H]'] 
    mol_alcohols = [Chem.MolFromSmiles(smiles) for smiles in alcohols_smiles]

    intermediates_ase = [rdkit_to_ase(intermediate) for intermediate in mol_alkanes + mol_alcohols]
    intermediates_formula = [intermediate.get_chemical_formula() for intermediate in intermediates_ase]
    intermediates_graph = [atoms_2_graph(intermediate, coords=True) for intermediate in intermediates_ase]
    intermediates_smiles = [Chem.MolToSmiles(intermediate) for intermediate in mol_alkanes + mol_alcohols]

    inter_precursor_dict = {}
    for smiles, formula, graph, ase_obj, rdkit_obj in zip(intermediates_smiles, intermediates_formula, intermediates_graph, intermediates_ase, mol_alkanes + mol_alcohols):
        inter_precursor_dict[smiles] = {
            'Smiles': smiles,
            'Formula': formula, 
            'Graph': graph, 
            'Atoms': ase_obj, 
            'RDKit': rdkit_obj,
            }

    isomeric_groups = id_group_dict(inter_precursor_dict)  # Define specific labels for isomers

    # 2) Generate all possible open-shell intermediates by H abstraction
    inter_dict, repeat_molec = {}, []
    for name, molec in isomeric_groups.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                molecule = inter_precursor_dict[molec_grp]['RDKit']
                intermediate = generate_pack(molecule, isomeric_groups[name].index(molec_grp) + 1)
                inter_dict[molec_grp] = intermediate
    
    # 3) Generate the connections between the intermediates via graph theory
    map_dict = {}
    for molecule in inter_dict.keys():
        intermediate = inter_dict[molecule]
        map_tmp = generate_map(intermediate, 'H')
        map_dict[molecule] = map_tmp

    return inter_dict, map_dict