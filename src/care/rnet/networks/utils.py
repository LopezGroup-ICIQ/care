import numpy as np
import multiprocessing as mp
import os

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
        s_keys = [inter.code for inter in inters] + [surf_inter.code] + [h_inter.code]
        stoic_dict = {key: 0 for key in s_keys}
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
        check_bonds = []
        for component in new_rxn.components:
            for inter in list(component):
                if inter.is_surface:
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
      
    h_inter = network_dict['[H][H]']['intermediates']['0001000101']    
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

    with mp.Pool(os.cpu_count()//2) as p:
        results = p.map(process_edge, args_list)
    
    for result in results:
        if result is not None:
            key, new_rxn = result
            network_dict[key]['reactions'].append(new_rxn)

    return network_dict


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
            stoic_dict = {surf_inter.code: -1, gas_inter.code: -1, inter.code: 1}     
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_inter]), frozenset([inter])), r_type='adsorption', stoic=stoic_dict))

    # H2 dissociative adsorption
    stoic_dict = {surf_inter.code: -1, gas_molecules['0002000101g'].code: -1, intermediates['0001000101'].code: 2}
    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_molecules['0002000101g']]), frozenset([intermediates['0001000101']])), r_type='adsorption', stoic=stoic_dict))
    # O2 dissociative adsorption
    stoic_dict = {surf_inter.code: -1, gas_molecules['0000020101g'].code: -1, intermediates['0000010101'].code: 2}
    adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, gas_molecules['0000020101g']]), frozenset([intermediates['0000010101']])), r_type='adsorption', stoic=stoic_dict))
    return gas_molecules, adsorption_steps

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