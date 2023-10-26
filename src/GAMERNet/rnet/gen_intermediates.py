from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from ase import Atoms, Atom
from itertools import product, combinations
import networkx as nx
import copy
from collections import defaultdict
import multiprocessing as mp

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

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

def rdkit_to_ase(rdkit_molecule: Chem.Mol) -> Atoms:
    """
    Generate an ASE Atoms object from an RDKit molecule.

    Parameters
    ----------
    rdkit_molecule : Chem.Mol
        RDKit molecule.

    Returns
    -------
    Atoms
        ASE Atoms object.
    """
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

def generate_alkanes_recursive(n_carbon, main_chain=""):
    if n_carbon == 0:
        return [main_chain]
    if n_carbon < 0:
        return []
    
    alkanes = []
    
    # Continue the main chain
    new_chain = main_chain + "C"
    alkanes += generate_alkanes_recursive(n_carbon-1, new_chain)
    
    # Add branches
    if len(main_chain) > 0:
        for i in range(len(main_chain)):
            if main_chain[i] == "C":
                new_chain = main_chain[:i] + "C(C)" + main_chain[i+1:]
                alkanes += generate_alkanes_recursive(n_carbon-1, new_chain)
    
    # Remove duplicates
    unique_alkanes = list(set(alkanes))
    return unique_alkanes

def canonicalize_smiles(smiles_list):
    canonical_smiles_set = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            canonical_smiles_set.add(canonical_smiles)
    return list(canonical_smiles_set)

def gen_alkanes_smiles(n):
    all_alkanes = []
    for i in range(1, n + 1):
        all_alkanes += generate_alkanes_recursive(i)
    unique_alkanes = canonicalize_smiles(all_alkanes)
    return unique_alkanes

def add_oxygens_to_molecule(mol: Chem.Mol, noc: int) -> set[str]:
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
    set[str]
        Set of all possible molecules with added oxygens (SMILES format).
    """
    unique_molecules = set()

    # Filter out carbon atoms with no free sites
    suitable_carbons = [(atom.GetIdx(), 4 - atom.GetDegree()) for atom in mol.GetAtoms() 
                        if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]
    if not suitable_carbons:
        return unique_molecules

    # Generate combinations
    combos = [list(range(num+1)) for _, num in suitable_carbons]

    for combo in product(*combos):
        total_oxygens = sum(combo)
        if total_oxygens == 0 or (noc >= 0 and total_oxygens > noc):
            continue

        tmp_mol = Chem.RWMol(mol)
        for num_oxygens, (carbon_idx, _) in zip(combo, suitable_carbons):
            for _ in range(num_oxygens):
                oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
                tmp_mol.AddBond(carbon_idx, oxygen_idx, order=Chem.rdchem.BondType.SINGLE)
        
        tmp_mol.UpdatePropertyCache()
        smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
        unique_molecules.add(smiles)

    return unique_molecules

def id_group_dict(molecules_dict: dict[str, dict]) -> dict[str, list[str]]:
    """
    Corrects the labeling for isomeric systems.

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

def rdkit_2_graph(mol: Chem.Mol) -> nx.Graph:
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
    
    nx_graph = nx.Graph()
    # Add atoms to graph
    for atom in mol.GetAtoms():
        nx_graph.add_node(atom.GetIdx(),
                   elem=atom.GetSymbol(),
                   xyz=conf.GetAtomPosition(atom.GetIdx()))
    
    # Add bonds to graph
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        pos_i = nx_graph.nodes[i]['xyz']
        pos_j = nx_graph.nodes[j]['xyz']
        
        length = pos_i.Distance(pos_j)
        
        nx_graph.add_edge(i, j, length=length)
    
    return nx_graph

def formula_from_rdkit(mol: Chem.Mol) -> str:
    """
    Get the molecular formula from an RDKit molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    str
        Chemical formula of the molecule as CxHyOz.
    """
    counts = defaultdict(int)
    
    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        counts[elem] += 1
    
    formula = ""
    
    for elem in ["C", "H", "O"]:
        count = counts.get(elem, 0)
        if count > 0:
            formula += f"{elem}{count}"
    
    return formula


def process_molecule(args: list) -> tuple[str, dict[str, MolPack]]:
    """
    Process a molecule to generate all the possible unique intermediates obtained from C-H bond breaking.
    Generates the pack dictionary for the molecule and its intermediates.

    Parameters
    ----------
    args : list
        List containing the name of the molecule, the molecule itself, the dictionary of intermediates, and the dictionary of isomeric groups.

    Returns
    -------
    tuple[str, dict]
        Tuple containing the name of the molecule and the dictionary of intermediates.
    """
    name, molec_grp, inter_precursor_dict, isomeric_groups = args
    molecule = inter_precursor_dict[molec_grp]['RDKit']
    intermediate = generate_pack(molecule, isomeric_groups[name].index(molec_grp) + 1)
    return molec_grp, intermediate

def process_intermediate(args: tuple[str, dict]) -> tuple[str, nx.DiGraph]:
    """
    Generates the map dictionary for the intermediate.
    The map dictionary is used to generate a primitive reaction network with only C-H bond breaking.

    Parameters
    ----------
    args : tuple
        Tuple containing the name of the molecule and the dictionary of intermediates.

    Returns
    -------
    tuple[str, nx.DiGraph]
        Tuple containing the name of the molecule and the nx.Digraph of intermediates.
    """
    molecule, intermediate = args
    map_tmp = generate_map(intermediate, 'H')
    return molecule, map_tmp

def generate_intermediates(n_carbon: int, n_oxy: int,) -> tuple[dict[str, dict[int, list[MolPack]]], dict[str, dict[int, nx.DiGraph]]]:
    """
    Generates all the possible intermediates for a given number of carbon atoms 
    starting from the set of fully saturated CHO molecules (alkanes, alcohols, etc.).

    Parameters
    ----------
    n_carbon : int
        Maximum number of carbon atoms in the intermediates.
    n_oxy : int
        Maximum number of oxygen atoms in the intermediates.

    Returns
    -------
    tuple[dict, dict]   
        Tuple containing the dictionary of intermediates and the dictionary of maps.     
    """
    
    # 1) Generate all closed-shell satuarated CHO molecules (alkanes and alcohols plus H2, H2O2 and O2)
    alkanes_smiles = gen_alkanes_smiles(n_carbon)    
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in list(alkanes_smiles)]
    
    alcohols_smiles = [add_oxygens_to_molecule(mol, n_oxy) for mol in mol_alkanes] 
    alcohols_smiles = [smiles for smiles_set in alcohols_smiles for smiles in smiles_set]  # flatten list of lists
    alcohols_smiles += ['CO','C(O)O','O', 'OO', '[H][H]'] 
    mol_alcohols = [Chem.MolFromSmiles(smiles) for smiles in alcohols_smiles]

    intermediates_formula = [formula_from_rdkit(intermediate) for intermediate in mol_alkanes + mol_alcohols]
    intermediates_graph = [rdkit_2_graph(intermediate) for intermediate in mol_alkanes + mol_alcohols]
    intermediates_smiles = [Chem.MolToSmiles(intermediate) for intermediate in mol_alkanes + mol_alcohols]

    inter_precursor_dict = {}
    for smiles, formula, graph, rdkit_obj in zip(intermediates_smiles, intermediates_formula, intermediates_graph, mol_alkanes + mol_alcohols):
        inter_precursor_dict[smiles] = {
            'Smiles': smiles,
            'Formula': formula, 
            'Graph': graph, 
            'RDKit': rdkit_obj,
            }
        
    isomeric_groups = id_group_dict(inter_precursor_dict)  # Define specific labels for isomers

    num_cores = mp.cpu_count()

    # 2) Generate all possible open-shell intermediates by H abstraction
    inter_dict, repeat_molec = {}, []

    args_list = []
    for name, molec in isomeric_groups.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                args_list.append([name, molec_grp, inter_precursor_dict, isomeric_groups])
    
    with mp.Pool(num_cores//2) as pool:
        results = pool.map(process_molecule, args_list)
    inter_dict = {k: v for k, v in results}

    # 3) Generate the connections between the intermediates via graph theory
    map_dict = {}
    args_map_list = [(molecule, inter_dict[molecule]) for molecule in inter_dict.keys()]
    with mp.Pool(num_cores//2) as pool:
        results = pool.map(process_intermediate, args_map_list)
    map_dict = {k: v for k, v in results}
    return inter_dict, map_dict