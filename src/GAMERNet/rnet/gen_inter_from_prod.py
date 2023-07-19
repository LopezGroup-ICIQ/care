import copy
from collections import namedtuple

import networkx as nx
from itertools import combinations
from ase import Atoms
from pubchempy import get_compounds, Compound
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

MolPack = namedtuple('MolPack',['code', 'mol','graph', 'subs'])

def generate_vars(input_molecule: str) -> tuple[str, Atoms, nx.Graph]:
    """Generates the molecular formula, ASE Atoms object and NetworkX Graph of a molecule.

    Parameters
    ----------
    input_molecule : str
        Name of the molecule.

    Returns
    -------
    tuple(str, Atoms, nx.Graph, Molecule)
        Molecular formula, ASE Atoms object and NetworkX Graph of the molecule.
    """

    pubchem_molecule = get_compounds(input_molecule, 'name', record_type='3d', listkey_count=1)[0]
    pubchem_atoms = [atom.element for atom in pubchem_molecule.atoms]
    pubchem_coordinates = [(atom.x, atom.y, atom.z) for atom in pubchem_molecule.atoms]
    molecule_ase_obj = Atoms(pubchem_atoms, positions=pubchem_coordinates)
    molecule_nx_graph = ase_coord_2_graph(molecule_ase_obj, coords=True)
    molecular_formula = molecular_formula_from_graph(molecule_nx_graph)

    return molecular_formula, molecule_ase_obj, molecule_nx_graph

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

def add_O_2_molec(molecule_graph: nx.Graph) -> nx.Graph:
    molec_graph_copy = copy.deepcopy(molecule_graph)
    molec_graph_copy.neighbors(0)
    # Detecting the carbon atoms
    carbon_nodes = []
    for node in molec_graph_copy.nodes():
        if molec_graph_copy.nodes[node]['elem'] == 'C':
            neighb = molec_graph_copy.neighbors(node)
            oxy_flag = False
            for n in neighb:
                if molec_graph_copy.nodes[n]['elem'] == 'O':
                    oxy_flag = True
                    break
            carbon_nodes.append((node, oxy_flag))

    # Checking if the carbon atoms are connected to a hydrogen atom
    for node, oxy_flg in carbon_nodes:
        if oxy_flg == True:
            return molec_graph_copy
        H_flag = False
        for neighbor in molec_graph_copy.neighbors(node):
            if molec_graph_copy.nodes[neighbor]['elem'] == 'H':
                while H_flag == False:
                    # Replacing the hydrogen atom with an oxygen atom
                    molec_graph_copy.nodes[neighbor]['elem'] = 'O'
                    H_flag = True
    
    return molec_graph_copy
     
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

def compare_strucures_from_pubchem(molecular_formula: str, saturated_molecule_graph: nx.Graph) -> tuple[nx.Graph, str, Atoms]:
    """Compares the molecular formula and graph of a molecule with the PubChem database to obtain the
    molecular formula, ASE Atoms object, and NetworkX Graph of the saturated molecule.

    Parameters
    ----------
    molecular_formula : str
        Molecular formula of the molecule.
    saturated_molecule_graph : nx.Graph
        NetworkX Graph of the saturated molecule.

    Returns
    -------
    tuple(nx.Graph, str, Atoms)
        Tuple containing the NetworkX Graph, molecular formula, ASE Atoms object of the saturated molecule.
    """
    
    pubchem_compounds = get_compounds(molecular_formula, 'formula', record_type='3d', listkey_count=20)
    for compound in pubchem_compounds:
        pubchem_cid = compound.cid
        c = Compound.from_cid(pubchem_cid)
        compound_formula = c.molecular_formula
        
        compound_atoms = [atom.element for atom in compound.atoms]
        compound_coords = [(atom.x, atom.y, atom.z) for atom in compound.atoms]

        compound_ase_obj = Atoms(compound_atoms, positions=compound_coords, pbc=True)
        compound_graph = ase_coord_2_graph(compound_ase_obj, coords=True)

        if nx.is_isomorphic(saturated_molecule_graph, compound_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
            compound_name = c.iupac_name
            if compound_name == 'oxidane':
                compound_name = 'water'
            return  compound_graph, compound_formula, compound_ase_obj, compound_name

def sat_H_graph(molecule_nx_graph: nx.Graph) -> tuple[nx.Graph, bool]:
    """For the unsaturated molecules, it adds the missing H atoms to the nx.Graphs.

    Parameters
    ----------
    molecule_nx_graph : nx.Graph
        NetworkX Graph representing the molecule.

    Returns
    -------
    nx.Graph
        NetworkX Graph representing the molecule with the added H atoms.

    Examples
    --------
    C -> CH4
    CH -> CH4
    C2H2 -> C2H6
    """

    n_nodes = molecule_nx_graph.number_of_nodes()
    node_conn = nx.degree(molecule_nx_graph)
    node_attrs = nx.nodes(molecule_nx_graph)

    max_conns = {'C': 4, 'O': 2}
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

# TODO: Redefine this function
def add_H_nodes(molecule_nx_graph: nx.Graph, name_molecule: str) -> tuple[nx.Graph, str, Atoms, str]:
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
    updated_molecular_formula, molecule_nx_graph, updated_ase_obj = None, None, None, 

    if unsat_flag:
        molecular_formula = molecular_formula_from_graph(copy_graph)
        molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule = compare_strucures_from_pubchem(molecular_formula, copy_graph)
    
    return molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule

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

def update_inter_dict(inter_dict: dict[str, dict[int, list]], input_init_dict: dict) -> dict:
    """Updates the intermediate dictionary to include the desired products as another entry in the dictionary.
    Parameters
    ----------
    inter_dict : dict
        Dictionary containing the intermediates for all the isomeric groups.
    input_init_dict : dict
        Dictionary containing the molecular formulas and graph for the all the case study systems.
    Returns
    -------
    dict
        Dictionary containing the intermediates for the isomeric groups with unsaturated molecules.
    """

    copy_dict = copy.deepcopy(inter_dict)
    unsat_molec = [molecule for molecule in input_init_dict.keys() if molecule != 'formic acid']

    another_copy = copy_dict
    for molecule in unsat_molec:
        old_graph = input_init_dict[molecule]['nx.Graph']
        _, _, _, updated_name  = add_H_nodes(old_graph, molecule)
        for molec in inter_dict.keys():
            if molec in updated_name:
                for sat_H_group in range(len(inter_dict[molec])):
                    for config in range(len(inter_dict[molec][sat_H_group])):
                        new_graph = inter_dict[molec][sat_H_group][config][2].to_undirected()

                        if nx.is_isomorphic(new_graph, old_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
                            system = inter_dict[molec][sat_H_group][config]
                            iso_idx = sat_H_group

                            for sat_h in inter_dict[molec]:
                                if sat_h < iso_idx:
                                    del copy_dict[molec][sat_h]
                            copy_dict[molec][iso_idx] = [system]

                            # another_copy = copy.deepcopy(inter_dict)
                            another_copy = copy.deepcopy(copy_dict)
                            another_copy[molec] = {}
                            idx_count = 0
                            for idx in copy_dict[molec].keys():
                                another_copy[molec][idx_count] = copy_dict[molec][idx]
                                idx_count += 1
                            value = another_copy.pop(molec)

                            # Update the dictionary with the new key
                            another_copy[molecule] = value
                            copy_dict = another_copy
    return another_copy

def gen_inter(input_molecule_list: list[str]) -> tuple[dict, dict]:
    
    product_list = copy.deepcopy(input_molecule_list)
    # Since we need to start from CO2, we need to add formic acid as an input molecule
    product_list.append('formic acid')
    
    input_dict = {}
    total_graph_list = []
    for input_molecule in product_list:
        molecular_formula, molecule_ase_obj, molecule_nx_graph = generate_vars(input_molecule)
        input_dict[input_molecule] = {
                'Formula': molecular_formula,
                'Atoms object': molecule_ase_obj,
                'nx.Graph': molecule_nx_graph,
            }

        oxy_molec_graph = add_O_2_molec(molecule_nx_graph)

        sat_oxy_res = add_H_nodes(oxy_molec_graph, input_molecule)
        sat_molecule_nx_graph_oxy = sat_oxy_res[0]

        # Adding new edge attribute to the graph where the bond type is defined and stored
        for edge in sat_molecule_nx_graph_oxy.edges():

            node_0 = sat_molecule_nx_graph_oxy.nodes[edge[0]]['elem']
            node_1 = sat_molecule_nx_graph_oxy.nodes[edge[1]]['elem']

            sat_molecule_nx_graph_oxy.edges[edge]['bond_type'] = f'{node_0}-{node_1}'

        breaking_bond = []
        for u, v, attr in sat_molecule_nx_graph_oxy.edges(data=True):
            if attr['bond_type'] == 'C-C':
                breaking_bond.append((u, v))
            elif attr['bond_type'] == 'C-O' or attr['bond_type'] == 'O-C':
                breaking_bond.append((u, v))

        subgraph_list = []
        for i in range(len(breaking_bond)-1):
            for comb in combinations(breaking_bond, i+1):
                n_subgraphs = len(comb) + 1
                copy_graph = copy.deepcopy(sat_molecule_nx_graph_oxy)
                for bond in comb:
                    copy_graph.remove_edge(bond[0], bond[1])
                for j in range(n_subgraphs):
                    subgraph = copy_graph.subgraph(list(nx.node_connected_component(copy_graph, list(nx.nodes(copy_graph))[j])))
                    # Resetting the indexes of the nodes for the subgraph
                    subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0, ordering='default', label_attribute=None)
                    subgraph_list.append(subgraph)
        
        for graph in subgraph_list:
            sat_graph, _ = sat_H_graph(graph)
            subgraph_list[subgraph_list.index(graph)] = sat_graph
            total_graph_list.append(sat_graph)
        
        # Adding the saturated oxygenated molecule to the list
        total_graph_list.append(sat_molecule_nx_graph_oxy)

    unique_graphs = []
    for graph in total_graph_list:
        is_duplicate = False
        for unique_graph in unique_graphs:
            if nx.is_isomorphic(graph, unique_graph,  node_match=lambda x, y: x["elem"] == y["elem"]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_graphs.append(graph)


    molec_formula = []
    for graph in unique_graphs:
        f = molecular_formula_from_graph(graph)
        molec_formula.append(f)

    graph_formula_tuple = list(zip(unique_graphs, molec_formula))

    inter_dict = {}
    for graph, formula in graph_formula_tuple:
        mol_graph, upd_formula, mol_ase_obj, iupac_name = compare_strucures_from_pubchem(formula, graph)
        inter_dict[iupac_name] = {
                'Name': iupac_name,
                'Formula': upd_formula,
                'Atoms object': mol_ase_obj,
                'nx.Graph': mol_graph,
            }
        
    id_group = id_group_dict(inter_dict)

    tmp_inter_dict = {}
    repeat_molec = []
    for name, molec in id_group.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                mol_ase_obj = inter_dict[molec_grp]['Atoms object']
                intermediate = fn.generate_pack(mol_ase_obj, sum(1 for atom in mol_ase_obj if atom.symbol == 'H'), id_group[name].index(molec_grp) + 1)
                tmp_inter_dict[molec_grp] = intermediate

    updated_inter_dict = update_inter_dict(tmp_inter_dict, input_dict)

    map_dict = {}
    for molecule in updated_inter_dict.keys():
        intermediate = updated_inter_dict[molecule]
        map_tmp = fn.generate_map(intermediate, 'H')
        map_dict[molecule] = map_tmp
    
    return updated_inter_dict, map_dict