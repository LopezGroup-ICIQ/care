import copy
from collections import namedtuple

import networkx as nx
from itertools import combinations
from ase import Atoms
from pubchempy import get_compounds
from GAMERNet.rnet.utilities import functions as fn
import GAMERNet.rnet.graphs.graph_fn as gfn


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
    molecule_nx_graph = gfn.ase_coord_2_graph(molecule_ase_obj, coords=True)
    molecular_formula = gfn.molecular_formula_from_graph(molecule_nx_graph)

    return molecular_formula, molecule_ase_obj, molecule_nx_graph


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
        _, _, _, updated_name  = gfn.add_H_nodes(old_graph, molecule)
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

        oxy_molec_graph = gfn.add_O_2_molec(molecule_nx_graph)

        # sat_oxy_res = sat_H_graph(oxy_molec_graph)
        sat_molecule_oxy = gfn.sat_H_graph(oxy_molec_graph)[0]
        sat_molecule_nx_graph_oxy = gfn.find_new_struct(sat_molecule_oxy)[0]

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
            sat_graph, _ = gfn.sat_H_graph(graph)
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
        f = gfn.molecular_formula_from_graph(graph)
        molec_formula.append(f)

    graph_formula_tuple = list(zip(unique_graphs, molec_formula))

    inter_dict = {}
    for graph, formula in graph_formula_tuple:
        mol_graph, upd_formula, mol_ase_obj, iupac_name = gfn.compare_strucures_from_pubchem(formula, graph)
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