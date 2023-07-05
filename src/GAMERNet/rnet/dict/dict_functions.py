# Description: Functions for dictionary generation and manipulation.
from graphs.graph_data import molecular_formula
from graphs.graph_generation import ase_coord_2_graph
import io_fns.io_functions as iofn
from ase.io import read
from pyRDTP.geomio import file_to_mol
import os
import networkx as nx
import itertools
import copy
import json as js

# Loading parameter file
with open('data/parameters.json') as f:
    params = js.load(f)

REJECT_DIR_STR = params['REJECT_DIR_STR']
ISO_LABEL = params['ISO_LABEL']
OLD_LABEL = params['OLD_LABEL']
ADD_H_LABEL = params['ADD_H_LABEL']

xyz_ext = ".xyz"

# Old gen_init_dict function
def formula_graph_dict(path: str) -> dict:
    """Generate a Python dictionary containing, for each molecule in the path, its molecular formula and associates NetworX Graph.
    Parameters
    ----------
    path : str
        Path to the folder containing the XYZ files of the systems.
    Returns
    -------
    dict
        Dictionary containing the molecular formula and nx.Graph of each molecule in the path
    """

    init_dict = {}
    files = iofn.gen_file_list(path)
    for file in files:
        if os.path.isdir(file):
            continue
        file_path = os.path.join(path, file)
        molecule = os.path.splitext(file)[0]
        ase_obj = read(file_path)
        nx_graph = ase_coord_2_graph(ase_obj, True)
        molec_formula = molecular_formula(nx_graph)
        init_dict[molecule] = {'formula': molec_formula, 'graph': nx_graph}
    
    return init_dict

def check_graph_isomorph_dict(data_dict: dict, path: str) -> dict:
    """Checks, through the isomorphism of the nx.Graphs, if there are any isomorphic molecules in the dictionary.
    Parameters
    ----------
    dict : dict
        Dictionary containing the molecular formulas and graphs for all the case study molecules.
    path : str
        Path to the folder containing the XYZ files of the systems.
    Returns
    -------
    dict
        Updated dictionary where duplicates have been removed.
    """
    molecule_list, graph_list = list(data_dict.keys()), [data['graph'] for data in data_dict.values()]

    # Generating the directory where the rejected molecules will be stored.
    reject_path = path + REJECT_DIR_STR
    os.makedirs(reject_path, exist_ok=True)
    
    # Checking isomorphism.
    for graph1, graph2 in itertools.combinations(graph_list, 2):
        if nx.is_isomorphic(graph1, graph2, node_match=lambda x, y: x["elem"] == y["elem"]):
            iso1 = molecule_list[graph_list.index(graph1)]
            iso2 = molecule_list[graph_list.index(graph2)]

            # Since the systems are isomorphic (the same molecule), we remove the one with the longest name.
            # This is helpful for systems that are saturated since their label is extended.
            rm_iso = iso1 if len(iso1) > len(iso2) else iso2

            # Updating files and dictionary.
            old_file = rm_iso + xyz_ext
            iso_file = rm_iso + ISO_LABEL + xyz_ext
            iofn.mv_file(path, old_file, reject_path, iso_file, slash=True)
            del data_dict[rm_iso]

    return data_dict

# Hybrid between graph and dict manipulation
def add_H_nodes(data_dict: dict) -> dict:
    """For the unsaturated molecules, it adds the missing H atoms to the nx.Graphs.
    This is done to ensure a label consistency between the saturated and unsaturated molecules.  
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the molecular formulas and graphs for all the case study molecules.
    Returns
    -------
    dict
        Updated dictionary where the unsaturated molecules have been updated with the missing H atoms.
    """
    cp_dict = copy.deepcopy(data_dict)
    for molecule, data in data_dict.items():
        graph = data['graph']
        n_node = graph.number_of_nodes()
        node_conn = nx.degree(graph)
        node_attrs = nx.nodes(graph)

        add_H_flag = False

        for node_idx, conn in node_conn:
            elem_node = dict(node_attrs)[node_idx]['elem']

            if (elem_node == 'C' and conn < 4) or (
                elem_node == 'O' and conn < 2
            ):
                add_H_flag = True
                idx_updt = node_idx

                cp_dict[molecule]['graph'].add_node(n_node, elem='H')
                cp_dict[molecule]['graph'].add_edge(idx_updt, n_node)
                cp_dict[molecule]['formula'] = molecular_formula(cp_dict[molecule]['graph'])

                n_node += 1
        
        if add_H_flag:
            cp_dict[molecule + ADD_H_LABEL] = cp_dict.pop(molecule)
        
        data_dict = cp_dict
    return data_dict

# Old check_graph_db_sat function
def compare_graphs_pubchem(data_dict: dict, path: str, TMP_DIR: str) -> None:
    """Compares the updated graphs with those generated from the PubChem files.
    If the graphs are isomorphic, it updates the file with the data from the PubChem compund.
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the molecular formulas and graphs for all the case study systems.
    path : str
        Input path where the files are located
    TMP_DIR : str
        Temporary path where the PubChem files are stored.
    UNSAT : str
        String that indicates if working with unsaturated molecules.
    """
    for molecule, data in data_dict.items():
        if ADD_H_LABEL in molecule:
            dict_graph = data['graph']
            for file in os.listdir(TMP_DIR):
                path_file = os.path.join(TMP_DIR, file)
                ase_obj = read(path_file)
                nx_graph = ase_coord_2_graph(ase_obj, True)
                if nx.is_isomorphic(nx_graph, dict_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
                    iofn.mv_file(TMP_DIR, file, path, molecule + xyz_ext, slash=True)
                    iofn.mv_file(path, molecule[:-len(ADD_H_LABEL)]+ xyz_ext, path+REJECT_DIR_STR, molecule[:-len(ADD_H_LABEL)] + OLD_LABEL + xyz_ext, slash=True)
                    break
    iofn.rm_dir(TMP_DIR)

# Old id_group function.
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
    combinations = itertools.combinations(molecules_dict.keys(), 2)

    for molec_1, molec_2 in combinations:
        # Ignores systems with different molecular formula.
        if molecules_dict[molec_1]['formula'] != molecules_dict[molec_2]['formula']:
            continue

        graph_1 = molecules_dict[molec_1]['graph']
        graph_2 = molecules_dict[molec_2]['graph']

        if not nx.is_isomorphic(graph_1, graph_2, node_match=lambda x, y: x["elem"] == y["elem"]):
            molec_dict.setdefault(molec_1, []).append(molec_1)
            molec_dict.setdefault(molec_1, []).append(molec_2)
            molec_dict.setdefault(molec_2, []).append(molec_1)
            molec_dict.setdefault(molec_2, []).append(molec_2)

    for molec in molecules_dict.keys():
        if molec not in molec_dict:
            molec_dict[molec] = [molec]

    return molec_dict

def molobj_dict(path: str, filelist: list) -> dict:
    """Generates a dictionary containing the pyRDTP.molecule.Molecule objects for the cas study systems.
    Parameters
    ----------
    path : str
        Directory path where the input XYZ files are located
    filelist : list
        List of of the files.
    Returns
    -------
    dict
        Dictionary containing the pyRDTP.molecule.Molecule objects for the cas study systems.
    """
    obj_dict = {}
    for file in filelist:
        filepath = path + '/' + file
        molec_id = os.path.splitext(file)[0]
        molec_obj = file_to_mol(filepath, 'xyz')
        obj_dict[molec_id] = molec_obj
    return obj_dict

def update_pack_dict(init_dict: dict, sat_dict: dict, pack_dict: dict) -> dict:
    """Updates the final dictionary and corrects the labels of the unsaturated systems in order to add consistency to the labeling.
    Parameters
    ----------
    init_dict : dict
        Initial dictionary containing the molecular formulas and graphs of all the case study systems prior to the update.
    sat_dict : dict
        Dictionary containing the molecular formulas and graphs of all case study systems after the update.
    pack_dict : dict
        Pack dict generated containing all the key information of interest
    Returns
    -------
    dict
        Updated version of Pack Dict with the proper labels
    """
    copy_dict = copy.deepcopy(pack_dict)
    for molec, data in sat_dict.items():
        if ADD_H_LABEL in molec:
            for mol, dat in init_dict.items():
                if mol in molec:
                    old_graph = dat['graph']
            
            for molecule, molec_data in pack_dict.items():
                if molecule in molec:
                    for sat_H_group in range(len(pack_dict[molecule])):
                        for config in range(len(pack_dict[molecule][sat_H_group])):
                            new_graph = (pack_dict[molecule][sat_H_group][config][2]).to_undirected()
                            if nx.is_isomorphic(new_graph, old_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
                                system = pack_dict[molecule][sat_H_group][config]
                                iso_idx = sat_H_group

                                for sat_h in pack_dict[molecule]:
                                    if sat_h < iso_idx:
                                        del copy_dict[molecule][sat_h]
                                copy_dict[molecule][iso_idx] = [system]

                                another_copy = copy.deepcopy(copy_dict)
                                idx_count = 0
                                for idx in copy_dict[molecule].keys():
                                    another_copy[molecule][idx_count] = copy_dict[molecule][idx]
                                    idx_count += 1
                                copy_dict = another_copy

    return another_copy

def remove_labels(dict):
    for molecule in dict.keys():
        if ADD_H_LABEL in molecule:
            dict[molecule[:-len(ADD_H_LABEL)]] = dict.pop(molecule)
    return dict

def mv_inp_files(path):
    for file in os.listdir(path):
        upd_path = path + '_updated'
        os.makedirs(upd_path, exist_ok=True)
        if file.endswith('_updated'+xyz_ext):
            iofn.mv_file(path, file, upd_path, file, slash=True)
            iofn.mv_file(path + REJECT_DIR_STR, file[:-(len(ADD_H_LABEL+xyz_ext))]+OLD_LABEL+xyz_ext, path, file[:-(len(ADD_H_LABEL+xyz_ext))]+xyz_ext, slash=True)
            