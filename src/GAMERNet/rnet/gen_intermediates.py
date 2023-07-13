import networkx as nx
from pubchempy import get_compounds, Compound
from ase import Atoms
import numpy as np
from itertools import product, combinations
from scipy.spatial import Voronoi
from pyRDTP.geomio import ASEAtoms, MolObj
from pyRDTP.molecule import Molecule
from GAMERNet.rnet.utilities import functions as fn
import copy


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
    cutoff = (CORDERO[element_i] + CORDERO[element_j]) + tolerance
    return cutoff

def get_voronoi_neighbourlist(atoms: Atoms, 
                              tolerance: float, 
                              ) -> np.ndarray:
    """Get the connectivity list from a Voronoi analysis, considering periodic boundary conditions (PBC).
    To have two atoms connected, these must satisfy two conditions:
    1. They must share a Voronoi facet.
    2. The distance between them must be less than the sum of their covalent radii (plus a tolerance).
    # Modified version of the original function provided by Santiago Morandi.

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

def ase_coord_2_graph(atoms: Atoms, coords: bool) -> nx.Graph:
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

def generate_vars(input_molecule: str) -> tuple[str, Atoms, nx.Graph, Molecule]:
    """Generates the molecular formula, ASE Atoms object, pyRDTP object and NetworkX Graph of a molecule.

    Parameters
    ----------
    input_molecule : str
        Name of the molecule.

    Returns
    -------
    tuple(str, Atoms, nx.Graph, Molecule)
        Molecular formula, ASE Atoms object, pyRDTP object and NetworkX Graph of the molecule.
    """

    pubchem_molecule = get_compounds(input_molecule, 'name', record_type='3d', listkey_count=1)[0]
    
    pubchem_atoms = [atom.element for atom in pubchem_molecule.atoms]
    pubchem_coordinates = [(atom.x, atom.y, atom.z) for atom in pubchem_molecule.atoms]
    molecule_ase_obj = Atoms(pubchem_atoms, positions=pubchem_coordinates)
    molecule_nx_graph = ase_coord_2_graph(molecule_ase_obj, coords=True)
    
    uni_coords = ASEAtoms(molecule_ase_obj).universal_convert()
    
    molecule_pyrdtp_obj = MolObj()
    molecule_pyrdtp_obj.universal_read(uni_coords)
    molecule_pyrdtp_obj = molecule_pyrdtp_obj.write()
    
    molecular_formula = molecular_formula_from_graph(molecule_nx_graph)

    return molecular_formula, molecule_ase_obj, molecule_nx_graph, molecule_pyrdtp_obj

def compare_strucures_from_pubchem(molecular_formula: str, saturated_molecule_graph: nx.Graph) -> tuple[nx.Graph, str, Atoms, Molecule]:
    """Compares the molecular formula and graph of a molecule with the PubChem database to obtain the
    molecular formula, ASE Atoms object, pyRDTP object and NetworkX Graph of the saturated molecule.

    Parameters
    ----------
    molecular_formula : str
        Molecular formula of the molecule.
    saturated_molecule_graph : nx.Graph
        NetworkX Graph of the saturated molecule.

    Returns
    -------
    tuple(nx.Graph, str, Atoms, Molecule)
        Tuple containing the NetworkX Graph, molecular formula, ASE Atoms object, pyRDTP object of the saturated molecule.
    """
    
    pubchem_compounds = get_compounds(molecular_formula, 'formula', record_type='3d', listkey_count=20)
    for compound in pubchem_compounds:
        compound_pyrdtp_obj = None
        pubchem_cid = compound.cid
        c = Compound.from_cid(pubchem_cid)
        compound_formula = c.molecular_formula
        compound_atoms = [atom.element for atom in compound.atoms]
        compound_coords = [(atom.x, atom.y, atom.z) for atom in compound.atoms]

        compound_ase_obj = Atoms(compound_atoms, positions=compound_coords)
        compound_graph = ase_coord_2_graph(compound_ase_obj, coords=True)

        uni_coords = ASEAtoms(compound_ase_obj).universal_convert()
        compound_pyrdtp_obj = MolObj()
        compound_pyrdtp_obj.universal_read(uni_coords)
        compound_pyrdtp_obj = compound_pyrdtp_obj.write()
        if nx.is_isomorphic(saturated_molecule_graph, compound_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
            return  compound_graph, compound_formula, compound_ase_obj, compound_pyrdtp_obj

def molecular_formula_from_graph(graph: nx.Graph) -> str:
    """Generates the molecular formula of a molecule from its graph representation.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX Graph representing the molecule.

    Returns
    -------
    str
        Molecular formula.
    """

    num_C, num_H, num_O = 0, 0, 0
    for node in graph.nodes(data=True):
        elem_node = node[1]['elem']
        if elem_node == 'C':
            num_C += 1
        elif elem_node == 'H':
            num_H += 1
        elif elem_node == 'O':
            num_O += 1
    # Generating the formula
    formula = ""
    if num_C > 0:
        formula += "C" + (str(num_C) if num_C > 1 else "")
    if num_H > 0:
        formula += "H" + (str(num_H) if num_H > 1 else "")
    if num_O > 0:
        formula += "O" + (str(num_O) if num_O > 1 else "")
    return str(formula)

def add_H_nodes(molecule_nx_graph: nx.Graph) -> nx.Graph:
    """For the unsaturated molecules, it adds the missing H atoms to the nx.Graphs.
    This is done to ensure a label consistency between the saturated and unsaturated molecules.
    Parameters
    ----------
    molecule_nx_graph : nx.Graph
        NetworkX Graph representing the molecule.
    Returns
    -------
    nx.Graph
        NetworkX Graph representing the molecule with the added H atoms.
    """
    copy_graph, unsat_flag = sat_H_graph(molecule_nx_graph)
    updated_molecular_formula, molecule_nx_graph, updated_ase_obj, updated_pyrdtp_obj = None, None, None, None

    if unsat_flag:
        molecular_formula = molecular_formula_from_graph(copy_graph)
        molecule_nx_graph, updated_molecular_formula, updated_ase_obj, updated_pyrdtp_obj = compare_strucures_from_pubchem(molecular_formula, copy_graph)
 
    return unsat_flag, molecule_nx_graph, updated_molecular_formula, updated_ase_obj, updated_pyrdtp_obj

def sat_H_graph(molecule_nx_graph):
    n_nodes = molecule_nx_graph.number_of_nodes()
    node_conn = nx.degree(molecule_nx_graph)
    node_attrs = nx.nodes(molecule_nx_graph)

    max_conns = {'C': 4, 'O': 2, 'N': 3, 'P': 5, 'S': 4}
    copy_graph = molecule_nx_graph.copy()
    unsat_flag = False

    for idx_node, conn in node_conn:
        elem_node = dict(node_attrs)[idx_node]['elem']
        for element, max_conn in max_conns.items():
            if elem_node == element:
                while conn < max_conn:
                    unsat_flag = True
                    idx_updt = idx_node
                    copy_graph.add_node(n_nodes, elem='H')
                    copy_graph.add_edge(idx_updt, n_nodes)
                    n_nodes += 1
                    conn += 1

    return copy_graph, unsat_flag

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
    """
    molec_dict = {}
    molec_combinations = combinations(molecules_dict.keys(), 2)

    for molec_1, molec_2 in molec_combinations:
        # Ignores systems with different molecular formula.
        if molecules_dict[molec_1]['Formula'] != molecules_dict[molec_2]['Formula']:
            continue

        graph_1 = molecules_dict[molec_1]['nx.Graph']
        graph_2 = molecules_dict[molec_2]['nx.Graph']

        if not nx.is_isomorphic(graph_1, graph_2, node_match=lambda x, y: x["elem"] == y["elem"]):
            molec_dict.setdefault(molec_1, []).append(molec_1)
            molec_dict.setdefault(molec_1, []).append(molec_2)
            molec_dict.setdefault(molec_2, []).append(molec_1)
            molec_dict.setdefault(molec_2, []).append(molec_2)

    for molec in molecules_dict.keys():
        if molec not in molec_dict:
            molec_dict[molec] = [molec]

    return molec_dict

def update_inter_dict(tmp_inter_dict: dict, input_init_dict: dict) -> dict:
    """Updates the intermediate dictionary to include only the isomeric groups with unsaturated molecules.
    Parameters
    ----------
    tmp_inter_dict : dict
        Dictionary containing the intermediates for all the isomeric groups.
    input_init_dict : dict
        Dictionary containing the molecular formulas and graph for the all the case study systems.
    Returns
    -------
    dict
        Dictionary containing the intermediates for the isomeric groups with unsaturated molecules.
    """

    copy_dict = copy.deepcopy(tmp_inter_dict)
    unsat_molec = [molecule for molecule, data in input_init_dict.items() if data['Unsaturated flag']]
    another_copy = copy_dict
    for molecule in unsat_molec:
        if input_init_dict[molecule]['Unsaturated flag']:
            old_graph = input_init_dict[molecule]['nx.Graph']

            for molec in tmp_inter_dict.keys():
                if molec in molecule:
                    for sat_H_group in range(len(tmp_inter_dict[molec])):
                        for config in range(len(tmp_inter_dict[molec][sat_H_group])):
                            new_graph = tmp_inter_dict[molec][sat_H_group][config][2].to_undirected()

                            if nx.is_isomorphic(new_graph, old_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
                                system = tmp_inter_dict[molec][sat_H_group][config]
                                iso_idx = sat_H_group

                                for sat_h in tmp_inter_dict[molec]:
                                    if sat_h < iso_idx:
                                        del copy_dict[molec][sat_h]
                                copy_dict[molec][iso_idx] = [system]

                                another_copy = copy.deepcopy(copy_dict)
                                another_copy[molec] = {}
                                idx_count = 0
                                for idx in copy_dict[molec].keys():
                                    another_copy[molec][idx_count] = copy_dict[molec][idx]
                                    idx_count += 1
                                copy_dict = another_copy

    return another_copy

def gen_intermediates(input_molecule_list):
    input_init_dict = {}
    for input_molecule in input_molecule_list:
        
        molecular_formula, molecule_ase_obj, molecule_nx_graph, molecule_pyrdtp_obj = generate_vars(input_molecule)
        unsat_flag, sat_molecule_nx_graph, updated_molecular_formula, updated_ase_obj, updated_pyrdtp_obj = add_H_nodes(molecule_nx_graph)

        input_init_dict[input_molecule] = {
            'Formula': molecular_formula,
            'Atoms object': molecule_ase_obj,
            'pyRDTP object': molecule_pyrdtp_obj,
            'nx.Graph': molecule_nx_graph,
            'Unsaturated flag': unsat_flag,
            'Updated formula': updated_molecular_formula,
            'Updated Atoms object': updated_ase_obj,
            'Updated pyRDTP object': updated_pyrdtp_obj,
            'Updated nx.Graph': sat_molecule_nx_graph,
        }

    id_group = id_group_dict(input_init_dict)

    tmp_inter_dict = {}
    repeat_molec = []
    for name, molec in id_group.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                if input_init_dict[molec_grp]['Unsaturated flag']:
                    updated_pyrdtp_obj = input_init_dict[molec_grp]['Updated pyRDTP object']
                else:
                    updated_pyrdtp_obj = input_init_dict[molec_grp]['pyRDTP object']

                intermediate = fn.generate_pack(updated_pyrdtp_obj, updated_pyrdtp_obj.elem_inf()['H'], id_group[name].index(molec_grp) + 1)
                tmp_inter_dict[molec_grp] = intermediate

    intermediate_dict = update_inter_dict(tmp_inter_dict, input_init_dict)

    map_dict = {}
    for molecule in intermediate_dict.keys():
        intermediate = intermediate_dict[molecule]
        map_tmp = fn.generate_map(intermediate, 'H')
        map_dict[molecule] = map_tmp

    return intermediate_dict, map_dict
