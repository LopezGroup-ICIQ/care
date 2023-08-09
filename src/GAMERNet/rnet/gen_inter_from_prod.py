from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from ase import Atoms
import pubchempy as pcp
from pubchempy import Compound
from itertools import combinations
from multiprocessing import Pool, cpu_count
import numpy as np
import networkx as nx
from GAMERNet.rnet.utilities import functions as fn


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

def matrix_to_mols(adj_matrix, mol):
    # Convert adjacency matrix to molecule
    new_mol = Chem.RWMol(mol)
    bond_indices = list(new_mol.GetBonds())

    for i, bond in enumerate(bond_indices):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if adj_matrix[begin_idx][end_idx] == 0 and adj_matrix[end_idx][begin_idx] == 0:
            new_mol.RemoveBond(begin_idx, end_idx)

    frags = Chem.GetMolFrags(new_mol, asMols=True)
    return [Chem.MolToSmiles(frag, canonical=True) for frag in frags]

def dfs(adj_matrix, mol, depth, seen, substructures):
    if depth == 0:
        return

    matrix_str = np.array2string(adj_matrix)
    if matrix_str in seen:
        return

    seen.add(matrix_str)

    subs = matrix_to_mols(adj_matrix, mol)
    substructures.update(subs)

    rows, cols = np.where(adj_matrix == 1)
    for i in range(len(rows)):
        row, col = rows[i], cols[i]
        if row < col:  # ensure we're not double-counting bonds
            adj_matrix[row][col] = 0
            adj_matrix[col][row] = 0
            dfs(np.copy(adj_matrix), mol, depth - 1, seen, substructures)
            adj_matrix[row][col] = 1
            adj_matrix[col][row] = 1

def is_linear(mol):
    """Check if a molecule is linear."""
    # Check if it's acyclic
    if rdmolops.GetSSSR(mol):
        return False

    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1:
            continue  # terminal atom
        elif atom.GetDegree() != 2:
            return False  # has branching
    return True

def get_linear_substructures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    substructures = set([smiles])  # Adding the original molecule to the set
    
    for bond in mol.GetBonds():
        # Create a copy of the molecule for bond breaking
        tmp_mol = Chem.RWMol(mol)
        
        # Break the bond
        tmp_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        # Get the two fragments as separate molecules
        fragments = Chem.GetMolFrags(tmp_mol, asMols=True)
        for frag in fragments:
            substructures.add(Chem.MolToSmiles(frag, canonical=True))
    
    return list(substructures)

def get_substructures_for_depth(args):
    smiles, depth = args
    mol = Chem.MolFromSmiles(smiles)
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    seen = set()
    substructures = set()

    dfs(adj_matrix, mol, depth, seen, substructures)
    return substructures

def get_all_substructures(smiles, n_jobs=cpu_count()):
    mol = Chem.MolFromSmiles(smiles)

    with Pool(n_jobs) as p:
        args = [(smiles, depth) for depth in range(mol.GetNumAtoms(), 0, -1)]
        results = p.map(get_substructures_for_depth, args)

    all_substructures = set()
    for substructures in results:
        all_substructures.update(substructures)

    return list(all_substructures)

def add_oxygens_to_molecule(mol):
    """Add oxygen to suitable carbon atoms in the molecule."""
    unique_molecules = set()

    # Identify suitable carbons with less than 4 bonds
    suitable_carbons = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]

    # For each suitable carbon, try adding an oxygen
    for r in range(1, len(suitable_carbons) + 1):
        for indices in combinations(suitable_carbons, r):
            tmp_mol = Chem.RWMol(mol)
            for idx in indices:
                # Add an oxygen and create a bond between the carbon and the oxygen
                oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
                tmp_mol.AddBond(idx, oxygen_idx, order=Chem.rdchem.BondType.SINGLE)
            # Convert to canonical SMILES for uniqueness
            smiles = Chem.MolToSmiles(tmp_mol, canonical=True)
            unique_molecules.add(smiles)

    return list(unique_molecules)

def process_substructures_chunk(substructures_chunk):
    modified_molecules = []
    for sub in substructures_chunk:
        mol = Chem.MolFromSmiles(sub)
        modified_molecules.extend(add_oxygens_to_molecule(mol))
    return modified_molecules

def generate_ase_obj(sub):
    mol = Chem.MolFromSmiles(sub, sanitize=True)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1)
    conformer_info = AllChem.UFFOptimizeMoleculeConfs(mol)
    conformer_energies = [item[1] for item in conformer_info]
    mol.GetConformer(int(np.argmin(conformer_energies)))
    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return Atoms(symbols=symbols, positions=positions)

def get_chemical_formula(atoms):
    symbols = atoms.get_chemical_symbols()
    counts = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1

    formula = ""
    for symbol, count in counts.items():
        formula += symbol
        if count > 1:
            formula += str(count)
    return formula

def ase_obj_2_graph(atoms: Atoms, coords: bool) -> nx.Graph:
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

def id_group_dict(molecules_dict: dict) -> dict:
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
        # Ignores systems with different molecular formula.
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




def gen_inter(carbon_backbone: str) -> tuple[dict, dict]:
    compound = carbon_backbone
    compound = pcp.get_compounds(compound, 'name')[0]
    pubchem_cid = compound.cid
    c = Compound.from_cid(pubchem_cid)
    # Generating the SMILES string
    smiles = c.canonical_smiles
    mol = Chem.MolFromSmiles(smiles)

    if is_linear(mol):
        all_substructures = get_linear_substructures(smiles)
    else:
        all_substructures = get_all_substructures(smiles)

    # Create a pool of workers (one per CPU core)
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    # Split the substructures into chunks for multiprocessing
    chunk_size = max(1, len(all_substructures) // num_processes)
    substructure_chunks = [all_substructures[i:i+chunk_size] for i in range(0, len(all_substructures), chunk_size)]

    # Process each chunk in parallel
    results = pool.map(process_substructures_chunk, substructure_chunks)
    
    # Combine results from all processes
    all_modified_molecules = [item for sublist in results for item in sublist]

    all_subs = all_substructures + all_modified_molecules + ['C(=O)O'] # Adding the formic acid to the list

    with Pool(cpu_count()) as p:
        all_atoms = p.map(generate_ase_obj, all_subs)
    
    # For each ase Atoms in the list, get the formula, and graph
    all_formulas = [get_chemical_formula(atoms) for atoms in all_atoms]
    all_graphs = [ase_obj_2_graph(atoms, coords=True) for atoms in all_atoms]

    # Create a dictionary of all the substructures
    all_substructures_dict = {}
    for smiles, formula, graph, ase_atoms in zip(all_subs, all_formulas, all_graphs, all_atoms):
        all_substructures_dict[smiles] = {
            'Formula': formula,
            'Graph': graph,
            'Atoms': ase_atoms
        }
    
    id_group = id_group_dict(all_substructures_dict)

    inter_dict = {}
    repeat_molec = []
    for name, molec in id_group.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                mol_ase_obj = all_substructures_dict[molec_grp]['Atoms']
                intermediate = fn.generate_pack(mol_ase_obj, sum(1 for atom in mol_ase_obj if atom.symbol == 'H'), id_group[name].index(molec_grp) + 1)
                inter_dict[molec_grp] = intermediate

    map_dict = {}
    for molecule in inter_dict.keys():
        intermediate = inter_dict[molecule]
        map_tmp = fn.generate_map(intermediate, 'H')
        map_dict[molecule] = map_tmp
    
    return inter_dict, map_dict