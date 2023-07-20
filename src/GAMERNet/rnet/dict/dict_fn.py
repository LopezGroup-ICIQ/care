import copy
from itertools import combinations
from ase import Atoms
import networkx as nx
from pubchempy import get_compounds
from collections import namedtuple
from GAMERNet.rnet.graphs import graph_fn as gfn

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
