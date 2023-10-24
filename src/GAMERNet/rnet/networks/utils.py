import numpy as np
import multiprocessing as mp
import os

import GAMERNet.rnet.utilities.additional_funcs as af
from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction

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
        if len(inters[0].molecule) > len(inters[1].molecule):
            rxn_lhs, rxn_rhs = (inters[0], surf_inter), (inters[1], h_inter)
        else:
            rxn_lhs, rxn_rhs = (inters[1], surf_inter), (inters[0], h_inter)
        new_rxn = ElementaryReaction(components=(rxn_lhs, rxn_rhs), r_type='C-H')
        hh_condition1 = 'C' not in inters[0].molecule.get_chemical_symbols() and 'C' not in inters[1].molecule.get_chemical_symbols()
        hh_condition2 = 'O' not in inters[0].molecule.get_chemical_symbols() and 'O' not in inters[1].molecule.get_chemical_symbols()
        if hh_condition1 and hh_condition2:
            new_rxn.r_type = 'H-H'
        check_bonds = []
        for component in new_rxn.components:
            for inter in list(component):
                if inter.is_surface or inter.code in ('01011-*', '02011-*'):
                    continue
                else:
                    oxy = [atom for atom in inter.molecule if atom.symbol == "O"]
                    if len(oxy) != 0:
                        counter = 0
                        for atom in oxy:
                            counter += np.count_nonzero(inter.molecule.arrays['conn_pairs'] == atom.index)
                        check_bonds.append(counter)
                    else:
                        check_bonds.append(0)
        if check_bonds[0] != check_bonds[1]:
            new_rxn.r_type = 'O-H'
        return (key, new_rxn)  # return a tuple or another data structure that suits you
    except KeyError:
        print("Key Error: Edge {} not found in the dictionary".format(edge))
        return None

def generate_network_dict(rxn_dict: dict, surf_inter: Intermediate) -> dict:
    """
    missing docstring
    """
    network_dict = {} 
    network_dict['[H][H]'] = {'intermediates': {}, 'reactions': []}
    for node in rxn_dict['[H][H]'].nodes():
        try:
            sel_node = rxn_dict['[H][H]'].nodes[node]
            new_inter = Intermediate(code=node+'*', 
                                        molecule=sel_node['mol'], 
                                        graph=sel_node['graph'],  
                                        phase='ads')
            network_dict['[H][H]']['intermediates'][node] = new_inter
        except KeyError:
            print("KeyError: Node {} not found in the dictionary".format(node))
            pass
      
    h_inter = network_dict['[H][H]']['intermediates']['01011']    
    args_list = []
    for key, network in rxn_dict.items():
        if key != '[H][H]':
            network_dict[key] = {'intermediates': {}, 'reactions': []}
            for node in network.nodes():
                try:
                    sel_node = network.nodes[node]
                    new_inter = Intermediate(code=node+'*', 
                                            molecule=sel_node['mol'], 
                                            graph=sel_node['graph'],  
                                            phase='ads')
                    network_dict[key]['intermediates'][node] = new_inter
                except KeyError:
                    print("KeyError: Node {} not found in the dictionary".format(node))
                    pass        
        for edge in list(network.edges()):
                args_list.append((key, edge, network_dict, h_inter, surf_inter))

    with mp.Pool(processes=os.cpu_count()) as p:
        results = p.map(process_edge, args_list)
    for result in results:
        if result is not None:
            key, new_rxn = result
            network_dict[key]['reactions'].append(new_rxn)
        # for edge in list(network.edges()):
        #     try:
        #         inters = (network_dict[key]['intermediates'][edge[0]], network_dict[key]['intermediates'][edge[1]])
        #         if len(inters[0].molecule) > len(inters[1].molecule):
        #             rxn_lhs, rxn_rhs = (inters[0], surf_inter), (inters[1], h_inter)
        #         else:
        #             rxn_lhs, rxn_rhs = (inters[1], surf_inter), (inters[0], h_inter)
        #         new_rxn = ElementaryReaction(components=(rxn_lhs, rxn_rhs), r_type='C-H')
        #         # classify H-H and O-H
        #         hh_condition1 = 'C' not in inters[0].molecule.get_chemical_symbols() and 'C' not in inters[1].molecule.get_chemical_symbols()
        #         hh_condition2 = 'O' not in inters[0].molecule.get_chemical_symbols() and 'O' not in inters[1].molecule.get_chemical_symbols()
        #         if hh_condition1 and hh_condition2:
        #             new_rxn.r_type = 'H-H'
        #         check_bonds = []
        #         for component in new_rxn.components:
        #             for inter in list(component):
        #                 if inter.is_surface or inter.code in ('01011-*', '02011-*'): #surface or a hydrogen atom,
        #                     continue
        #                 else:
        #                     oxy = [atom for atom in inter.molecule if atom.symbol == "O"]
        #                     if len(oxy) != 0:
        #                         counter = 0
        #                         for atom in oxy:
        #                             counter += np.count_nonzero(inter.molecule.arrays['conn_pairs'] == atom.index)
        #                         check_bonds.append(counter)
        #                     else:
        #                         check_bonds.append(0)
        #         if check_bonds[0] != check_bonds[1]:
        #             new_rxn.r_type = 'O-H'
                
        #         for reaction in network_dict[key]['reactions']:
        #             if reaction.components == new_rxn.components:
        #                 break
        #         else: # if the for loop is not broken
        #             network_dict[key]['reactions'].append(new_rxn)
        #     except KeyError:
        #         print("Key Error: Edge {} not found in the dictionary".format(edge))
        #         pass
    return network_dict

# def add_energies_to_dict(network_dict, energ_entr_dict):
#     """
#     missing docstring
#     """
#     for net in network_dict.values():
#         for code, inter in net['intermediates'].items():
#             test = [n for n in energ_entr_dict.values()]
#             for i in test:
#                 for vals in i.values():
#                     # try:
#                     #     inter.energy = vals[code]['e_zpe_solv']
#                     #     inter.entropy = vals[code]['S_meV']
#                     # except KeyError:
#                     inter.energy = 0.0
#                     inter.entropy = 0.0
#                         # pass
#     return network_dict

def gen_adsorption_reactions(intermediates: dict[str, Intermediate], surf_inter: Intermediate) -> tuple[dict[str, Intermediate], list[ElementaryReaction]]:
    """
    Generate all Intermediates that can desorb from the surface and the corresponding desorption reactions, including
    the dissociative adsorption of H2 and O2.
    """
    adsorption_steps = []
    gas_molecules = {}
    for inter in intermediates.values():
        if inter.closed_shell:
            gas_code = inter.code[:-1] + 'g'
            gas_inter = Intermediate.from_molecule(inter.molecule, code=gas_code, phase='gas')
            gas_molecules[gas_code] = gas_inter           
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_inter]), frozenset([inter])), r_type='adsorption'))

    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_molecules['02011g']]), frozenset([intermediates['01011']])), r_type='adsorption'))
    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_molecules['00211g']]), frozenset([intermediates['00111']])), r_type='adsorption'))
    return gas_molecules, adsorption_steps


def classify_oh_bond_breaks(reactions_list: list[ElementaryReaction]) -> None:
    """
    Classify elementary reactions with O-H bond-breaking
    It does not add any reaction to the network. It just classifies the reactions

    Parameters
    ----------
    reactions_list : list[ElementaryReaction]
        List of elementary reactions.

    Returns
    -------
    None
    """

    for reaction in reactions_list:
        check_bonds = []
        if reaction.r_type == 'C-H':
            for state in reaction.components:
                for inter in list(state):
                    if inter.is_surface or inter.code in ('01011-*', '02011-*'): #surface or a hydrogen atom,
                        continue
                    else:
                        oxy = [atom for atom in inter.molecule if atom.symbol == "O"]
                        if len(oxy) != 0:
                            counter = 0
                            for atom in oxy:
                                counter += np.count_nonzero(inter.molecule.arrays['conn_pairs'] == atom.index)
                            check_bonds.append(counter)
                        else:
                            check_bonds.append(0)
            if check_bonds[0] != check_bonds[1]:
                reaction.r_type = 'O-H'


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

# def gen_gas_dict(gas_df):
#     """
#     missing docstring
#     """
#     gas_dict = {gas_code: {'mol': None, 'conn_graph': None, 'energy': None, 
#                        'entropy': None, 'electrons': None} for gas_code in gas_df.gas}


#     for gas_code in gas_df.gas:
#         gas_data = gas_df.loc[gas_df['gas'] == gas_code]

#         gas_mol = pickle.loads(gas_data.contcar_object.values[0])
#         gas_dict[gas_code]['mol'] = gas_mol

#         gas_graph = pickle.loads(gas_data.conn_graph.values[0])
#         gas_dict[gas_code]['conn_graph'] = gas_graph

#         gas_dict[gas_code]['energy'] = gas_data.e_zpe.values[0]
#         gas_dict[gas_code]['entropy'] = gas_data.S_meV.values[0]

#         if np.isnan(gas_dict[gas_code]['energy']):
#             gas_dict[gas_code]['energy'] = 0.0
#         if np.isnan(gas_dict[gas_code]['entropy']):
#             gas_dict[gas_code]['entropy'] = 0.0

#         electrons = af.adjust_electrons(gas_mol)
#         gas_dict[gas_code]['electrons'] = electrons
    
#     return gas_dict

# def gen_gas_inter_dict(gas_dict: dict) -> dict:
#     """
#     missing docstring
#     """
#     gas_inter_dict = {}
#     for gas_code, gas_data in gas_dict.items():
#         new_inter = Intermediate(gas_code, molecule=gas_data['mol'], energy=gas_data['energy'], 
#                                 entropy=gas_data['entropy'], electrons=gas_data['electrons'], phase='gas')
#         gas_inter_dict[gas_code] = new_inter
#     return gas_inter_dict