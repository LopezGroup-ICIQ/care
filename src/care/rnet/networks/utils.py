import numpy as np
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations
from ase import Atoms

from ase.neighborlist import neighbor_list
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.elementary_reaction import ElementaryReaction
from care.rnet.utilities.functions import MolPack

metal_structure_dict = {
    "Ag": "fcc",
    "Au": "fcc",
    "Cd": "hcp",
    "Co": "hcp",
    "Cu": "fcc",
    "Fe": "bcc",
    "Ir": "fcc",
    "Ni": "fcc",
    "Os": "hcp",
    "Pd": "fcc",
    "Pt": "fcc",
    "Rh": "fcc",
    "Ru": "hcp",
    "Zn": "hcp"
}

#### OLD FUNCTIONS ####

# def generate_dict(inter_dict: dict) -> dict:
#     """
#     missing docstring
#     """
#     lot1_att = {}
#     for network, subs in inter_dict.items(): #reading the pickle file and making a dictionary
#         lot1_att[network] = {}
#         tmp_dict = lot1_att[network]
#         for mol_lst in subs.values():
#             for inter in mol_lst:
#                 tmp_dict[inter.code] = {'mol': inter.ase_mol, 'graph': inter.nx_graph}
#     return lot1_att

# def process_edge(args: tuple):
#     """
#     Process an edge of the reaction network and generate the corresponding ElementaryReaction object.

#     Parameters
#     ----------
#     args : tuple
#         Tuple containing the key of the reaction network, the edge to process, the dictionary containing the intermediates of the reaction network, and the intermediates of the reaction network corresponding to the hydrogen and surface species.

#     Returns
#     -------
#     tuple
#         Tuple containing the key of the reaction network and the ElementaryReaction object corresponding to the edge.
#     """
#     key, edge, network_dict, h_inter, surf_inter = args
#     try:
#         inters = (network_dict[key]['intermediates'][edge[0]], network_dict[key]['intermediates'][edge[1]])
#         stoic_dict = defaultdict(int)
#         if len(inters[0].molecule) > len(inters[1].molecule):
#             rxn_lhs, rxn_rhs = (inters[0], surf_inter), (inters[1], h_inter)
#         else:
#             rxn_lhs, rxn_rhs = (inters[1], surf_inter), (inters[0], h_inter)
#         for inter in rxn_lhs:
#             stoic_dict[inter.code] -= 1
#         for inter in rxn_rhs:
#             stoic_dict[inter.code] += 1
#         new_rxn = ElementaryReaction(components=(rxn_lhs, rxn_rhs), r_type='C-H', stoic=stoic_dict)
#         hh_condition1 = 'C' not in inters[0].molecule.get_chemical_symbols() and 'C' not in inters[1].molecule.get_chemical_symbols()
#         hh_condition2 = 'O' not in inters[0].molecule.get_chemical_symbols() and 'O' not in inters[1].molecule.get_chemical_symbols()
#         if hh_condition1 and hh_condition2:
#             new_rxn.r_type = 'H-H'
#         if inters[0]['O'] != 0:
#             check_bonds = []
#             for component in new_rxn.components:
#                 for inter in list(component):
#                     oxy_indices = [atom.index for atom in inter.molecule if atom.symbol == "O"]
#                     if len(oxy_indices) != 0:
#                         counter = 0
#                         for atom_index in oxy_indices:
#                             counter += np.count_nonzero(inter.molecule.arrays['conn_pairs'] == atom_index)
#                         check_bonds.append(counter)
#                     else:
#                         check_bonds.append(0)
#             new_rxn.r_type = 'H-O' if check_bonds[0] != check_bonds[1] else 'C-H'
#         return (key, new_rxn)  
#     except KeyError:
#         print("Key Error: Edge {} not found in the dictionary".format(edge))
#         return None

############################################################################################

def gen_inter_objs(inter_dict: dict[str, dict[int, list[MolPack]]]) -> dict:
    """
    Generates a dictionary containing the Intermediate instances of all the chemical species of the reaction network.
    If the molecule is closed-shell, the corresponding Intermediate instance is also generated.

    Parameters
    ----------
    inter_dict : dict
        Dictionary containing the intermediates of the reaction network.
        each key is the smiles of a saturated CHO molecule, and each value is a dictionary   
        containing the intermediates of the reaction network for that molecule.
        Each key of the subdictionary is the number of hydrogen removed from the molecule, 
        and each value is a list of MolPack objects, each one representing a specific isomer.

    Returns
    -------
    network_dict: dict
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.
    """
    network_dict = {} 
    for key, values in inter_dict.items():
        network_dict[key] = {'intermediates': {}}
        for value_list in values.values():
            for value in value_list:
                new_inter = Intermediate(code=value[0]+'*',
                                        molecule=value[1],
                                        graph=value[2],
                                        phase='ads')
                network_dict[key]['intermediates'][value.code+"*"] = new_inter
                if new_inter.closed_shell:
                    new_inter_gas = Intermediate(code=value[0]+'g',
                                        molecule=value[1],
                                        graph=value[2],
                                        phase='gas')
                    network_dict[key]['intermediates'][value.code+'g'] = new_inter_gas        
    return network_dict

def gen_adsorption_reactions(intermediates: dict[str, Intermediate]) -> list[ElementaryReaction]:
    """
    Generate all Intermediates that can desorb from the surface and the corresponding desorption reactions, including
    the dissociative adsorption of H2 and O2.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

    Returns
    -------
    adsorption_steps: list[ElementaryReaction]
        List of all desorption reactions.
    """
    adsorption_steps = []
    surf_inter = intermediates['0000000000*']
    for inter in intermediates.values():
        if inter.phase == 'gas':
            ads_inter = intermediates[inter.code[:-1]+'*']
            stoic_dict = {surf_inter.code: -1, inter.code: -1, ads_inter.code: 1}     
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, inter]), frozenset([ads_inter])), r_type='adsorption', stoic=stoic_dict))

    # dissociative adsorptions for H2 and O2
    for molecule in ['0002000101', '0000020101']:  
        stoic_dict = {surf_inter.code: -2, intermediates[molecule+'g'].code: -1, intermediates[molecule.replace("2", "1")+'*'].code: 2}
        adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates[molecule+'g']]), frozenset([intermediates[molecule.replace("2", "1")+'*']])), r_type='adsorption', stoic=stoic_dict))
    return adsorption_steps

def gen_rearrangement_reactions(intermediates: dict[str, Intermediate]) -> list[ElementaryReaction]:
    """
    Generate all 1,2-rearrangement reactions involving hydrogen atoms.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

    Returns
    -------
    rearrengement_steps: list[ElementaryReaction]
        List of all 1,2-rearrangement reactions involving hydrogen atoms.
    """
    rearrangement_steps = []
    inter_dict = defaultdict(list)
    for inter in intermediates.values():
        if inter.is_surface or inter.phase == 'gas':
            continue
        inter_dict[inter.code[:8]].append(inter)
    for value in inter_dict.values():
        if len(value) == 1:
            continue
        rearrangement_pairs = [[inter_1, inter_2] for inter_1, inter_2 in combinations(value, 2) if is_hydrogen_rearranged(inter_1.molecule, inter_2.molecule)]
        if rearrangement_pairs:
            for pair in rearrangement_pairs:
                code_1 = pair[0].code
                code_2 = pair[1].code
            stoic_dict = {code_1: -1, code_2: 1}            
            rearrangement_steps.append(ElementaryReaction(components=(frozenset([intermediates[code_1]]), frozenset([intermediates[code_2]])), r_type='rearrangement', stoic=stoic_dict))
    return rearrangement_steps

def is_hydrogen_rearranged(molecule_1: Atoms, molecule_2: Atoms):
    """
    Check for two molecules if there are potential hydrogen rearrangements.

    Parameters
    ----------
    molecule_1 : Atoms
        First molecule.
    molecule_2 : Atoms
        Second molecule.

    Returns
    -------
    bool
        True if there is a potential hydrogen rearrangement between the two molecules, False otherwise.
    """
    # Get indices of C, H, and O atoms
    c_indices = [atom.index for atom in molecule_1 if atom.symbol == 'C']
    h_indices = [atom.index for atom in molecule_1 if atom.symbol == 'H']
    o_indices = [atom.index for atom in molecule_1 if atom.symbol == 'O']
   
    # Defining cutoff for neighbor list
    cutoff = {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'O'): 1.5, ('O', 'O'): 1.5, ('O', 'H'): 1.3}
    # Calculate neighbor lists for each Atoms object
    n_list_1 = neighbor_list('ijS', molecule_1, cutoff=cutoff)
    n_list_2 = neighbor_list('ijS', molecule_2, cutoff=cutoff)
   
    # Convert neighbor lists to sets of tuples (atom index, neighbor index)
    bonds_1 = set(zip(n_list_1[0], n_list_1[1]))
    bonds_2 = set(zip(n_list_2[0], n_list_2[1]))
   
    # Check if the connectivity for C and O atoms is the same
    if not all((c, o) in bonds_2 for c in c_indices for o in o_indices if (c, o) in bonds_1):
        return False
   
    # Check for rearrangement of H atoms
    rearranged_h = []
    for h_index in h_indices:
        # Neighbors in both Atoms objects
        neighbors_1 = {j for i, j in bonds_1 if i == h_index}
        neighbors_2 = {j for i, j in bonds_2 if i == h_index}
       
        # If the H atom has different neighbors, and the different neighbor is a C or O atom
        if neighbors_1 != neighbors_2 and any((n in c_indices or n in o_indices) for n in neighbors_1.symmetric_difference(neighbors_2)):
            rearranged_h.append(h_index)
   
    # Only one H atom should have different connectivity to be a rearrangement reaction
    return len(rearranged_h) == 1


# def ts_energies(ts_states: list, neb_dict: dict, neb_df, surf_inter) -> None:
#     """
#     missing docstring
#     """
#     for reaction in ts_states:
#         reaction.energy = 0.0
#         reaction.entropy = 0.0
#         if reaction.r_type == 'C-C':
#             fs_components = reaction.components[0]
#             is_components = reaction.components[1]

#             for molecule in fs_components:
#                 if molecule.code == '0-0-0-0-0-*':
#                     continue
#                 else:
#                     fs_graph = molecule.graph

#             is_graphs = []
#             for molecule in is_components:
#                 if molecule.code == '0-0-0-0-0-*':
#                     continue
#                 else:
#                     is_graphs.append(molecule.graph)
#             possible_rxns = []
#             for rxn, ts_data in neb_dict.items():
#                 try:
#                     if nx.is_isomorphic(fs_graph, ts_data['fs']['conn_graph'][0], node_match=lambda x, y: x["elem"] == y["elem"]):
#                         fs_system = ts_data['fs']['conn_graph'][0]
#                         possible_rxns.append(rxn)
#                 except TypeError:
#                     continue

#             for poss_rxn in possible_rxns:
#                 reaction_db = neb_dict.get(poss_rxn)
            
#                 dict_graphs = (reaction_db.get('is').get('conn_graph'))
#                 for comb in it.product(dict_graphs, is_graphs):
#                     if nx.is_isomorphic(comb[0], comb[1], node_match=lambda x, y: x["elem"] == y["elem"]):
#                         # Getting the energies from neb_db
#                         try:
                            
#                             zpe_energ = neb_df.loc[neb_df['rxn'] == poss_rxn].e_zpe_solv.values[0]
#                             entropy = neb_df.loc[neb_df['rxn'] == poss_rxn].S_meV.values[0]
#                             zpe_energ_surf = zpe_energ + surf_inter.energy
#                             if np.isnan(zpe_energ):
#                                 zpe_energ_surf = 0.0
#                             if np.isnan(entropy):
#                                 entropy = 0.0
#                             reaction.energy = zpe_energ_surf
#                             reaction.entropy = entropy
#                         except:
#                             continue
#                         break