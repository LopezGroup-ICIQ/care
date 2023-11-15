import numpy as np
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations

from ase.neighborlist import neighbor_list
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.elementary_reaction import ElementaryReaction

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

def generate_dict(inter_dict: dict) -> dict:
    """
    missing docstring
    """
    lot1_att = {}
    for network, subs in inter_dict.items(): #reading the pickle file and making a dictionary
        lot1_att[network] = {}
        tmp_dict = lot1_att[network]
        for mol_lst in subs.values():
            for inter in mol_lst:
                tmp_dict[inter.code] = {'mol': inter.ase_mol, 'graph': inter.nx_graph}
    return lot1_att

def process_edge(args):
    key, edge, network_dict, h_inter, surf_inter = args
    try:
        inters = (network_dict[key]['intermediates'][edge[0]], network_dict[key]['intermediates'][edge[1]])
        stoic_dict = defaultdict(int)
        if len(inters[0].molecule) > len(inters[1].molecule):
            rxn_lhs, rxn_rhs = (inters[0], surf_inter), (inters[1], h_inter)
        else:
            rxn_lhs, rxn_rhs = (inters[1], surf_inter), (inters[0], h_inter)
        for inter in rxn_lhs:
            stoic_dict[inter.code] -= 1
        for inter in rxn_rhs:
            stoic_dict[inter.code] += 1
        new_rxn = ElementaryReaction(components=(rxn_lhs, rxn_rhs), r_type='C-H', stoic=stoic_dict)
        hh_condition1 = 'C' not in inters[0].molecule.get_chemical_symbols() and 'C' not in inters[1].molecule.get_chemical_symbols()
        hh_condition2 = 'O' not in inters[0].molecule.get_chemical_symbols() and 'O' not in inters[1].molecule.get_chemical_symbols()
        if hh_condition1 and hh_condition2:
            new_rxn.r_type = 'H-H'
        if inters[0]['O'] != 0:
            check_bonds = []
            for component in new_rxn.components:
                for inter in list(component):
                    oxy_indices = [atom.index for atom in inter.molecule if atom.symbol == "O"]
                    if len(oxy_indices) != 0:
                        counter = 0
                        for atom_index in oxy_indices:
                            counter += np.count_nonzero(inter.molecule.arrays['conn_pairs'] == atom_index)
                        check_bonds.append(counter)
                    else:
                        check_bonds.append(0)
            new_rxn.r_type = 'H-O' if check_bonds[0] != check_bonds[1] else 'C-H'
        return (key, new_rxn)  
    except KeyError:
        print("Key Error: Edge {} not found in the dictionary".format(edge))
        return None

def generate_network_dict(inter_dict: dict) -> dict:
    """
    Generates all the intermediates in which the bond-breaking is between a H atom and any other atom.
    """
    network_dict = {} 
    for key, values in inter_dict.items():
        network_dict[key] = {'intermediates': {}}
        for value_list in values.values():
            try:
                for value in value_list:
                    new_inter = Intermediate(code=value[0]+'*',
                                            molecule=value[1],
                                            graph=value[2],
                                            phase='ads')
                    network_dict[key]['intermediates'][value.code] = new_inter
                    if new_inter.closed_shell:
                        new_inter_gas = Intermediate(code=value[0]+'g',
                                            molecule=value[1],
                                            graph=value[2],
                                            phase='gas')
                        network_dict[key]['intermediates'][value.code+'g'] = new_inter_gas
            except KeyError:
                pass        
    return network_dict


def gen_adsorption_reactions(intermediates: dict[str, Intermediate], surf_inter: Intermediate) -> tuple[dict[str, Intermediate], list[ElementaryReaction]]:
    """
    Generate all Intermediates that can desorb from the surface and the corresponding desorption reactions, including
    the dissociative adsorption of H2 and O2.
    """
    adsorption_steps = []
    for inter in intermediates.values():
        if inter.phase == 'gas':

            stoic_dict = {surf_inter.code: -1, inter.code: -1, inter.code: 1}     
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, inter]), frozenset([inter])), r_type='adsorption', stoic=stoic_dict))

    # H2 dissociative adsorption
    stoic_dict = {surf_inter.code: -1, intermediates['0002000101g'].code: -1, intermediates['0001000101'].code: 2}
    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates['0002000101g']]), frozenset([intermediates['0001000101']])), r_type='adsorption', stoic=stoic_dict))
    # O2 dissociative adsorption
    stoic_dict = {surf_inter.code: -1, intermediates['0000020101g'].code: -1, intermediates['0000010101'].code: 2}
    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates['0000020101g']]), frozenset([intermediates['0000010101']])), r_type='adsorption', stoic=stoic_dict))
    return adsorption_steps


def gen_rearrangement_reactions(inter_dict, intermediates: dict[str, Intermediate]) -> list[ElementaryReaction]:
    """_summary_

    Parameters
    ----------
    inter_dict : _type_
        _description_
    surf_inter : _type_
        _description_

    Returns
    -------
    list[ElementaryReaction]
        _description_
    """
    rearrangement_steps = []
    for molecule, data in inter_dict.items():
        for remvd_H, molpack_list in data.items():
            # Getting the graphs from the molpack list
            ase_mol = [molpack[1] for molpack in molpack_list]
            # Generating H rearrangement reactions
            rearrangement_pairs = [[mol1, mol2] for mol1, mol2 in combinations(ase_mol, 2) if is_hydrogen_rearranged(mol1, mol2)]
            # Checking if the rearrangement pairs are valid (i.e., if the rearrangement goes from one atom to its direct neighbor)
            if rearrangement_pairs:
                # Getting the code of the molecule
                for pair in rearrangement_pairs:
                    code1 = molpack_list[ase_mol.index(pair[0])][0]
                    code2 = molpack_list[ase_mol.index(pair[1])][0]
                stoic_dict = {intermediates[code1].code: -1, intermediates[code2].code: 1}
                
                rearrangement_steps.append(ElementaryReaction(components=(frozenset([intermediates[code1]]), frozenset([intermediates[code2]])), r_type='rearrangement', stoic=stoic_dict))
    return rearrangement_steps

def is_hydrogen_rearranged(atoms1, atoms2, cutoff=1.2):
    """
    Check if two ASE Atoms objects represent a rearrangement reaction with one H atom moving.
    The cutoff parameter is used to determine if two atoms are neighbors (bonded).
    """
    # Get indices of C, H, and O atoms
    c_indices = [atom.index for atom in atoms1 if atom.symbol == 'C']
    h_indices = [atom.index for atom in atoms1 if atom.symbol == 'H']
    o_indices = [atom.index for atom in atoms1 if atom.symbol == 'O']
    
    # Calculate neighbor lists for each Atoms object
    n_list1 = neighbor_list('ijS', atoms1, cutoff=cutoff)
    n_list2 = neighbor_list('ijS', atoms2, cutoff=cutoff)
    
    # Convert neighbor lists to sets of tuples (atom index, neighbor index)
    bonds1 = set(zip(n_list1[0], n_list1[1]))
    bonds2 = set(zip(n_list2[0], n_list2[1]))
    
    # Check if the connectivity for C and O atoms is the same
    if not all((c, o) in bonds2 for c in c_indices for o in o_indices if (c, o) in bonds1):
        return False
    
    # Check for rearrangement of H atoms
    rearranged_h = []
    for h_index in h_indices:
        # Neighbors in both Atoms objects
        neighbors1 = {j for i, j in bonds1 if i == h_index}
        neighbors2 = {j for i, j in bonds2 if i == h_index}
        
        # If the H atom has different neighbors, and the different neighbor is a C or O atom
        if neighbors1 != neighbors2 and any((n in c_indices or n in o_indices) for n in neighbors1.symmetric_difference(neighbors2)):
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