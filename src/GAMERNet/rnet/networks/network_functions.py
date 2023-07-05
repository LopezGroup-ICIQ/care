import GAMERNet.rnet.utilities.additional_funcs as af
from GAMERNet.rnet.networks.networks import Intermediate, TransitionState
import networkx as nx
import itertools as it
import pickle
import numpy as np


def generate_dict(lot1):
    lot1_att = {}
    for network, subs in lot1.items(): #reading the pickle file and making a dictionary
        lot1_att[network] = {}
        tmp_dict = lot1_att[network]
        for mol_lst in subs.values():
            for inter in mol_lst:
                tmp_dict[inter.code] = {'mol': inter.mol, 'graph': inter.graph, 'energy': 0, 'entropy':0}

    return lot1_att

def generate_network_dict(map1, surf_inter, h_inter):
    network_dict = {} 
    for key, network in map1.items():
        network_dict[key] = {'intermediates': {}, 'ts': []}
        for node in network.nodes():
            try:
                sel_node = network.nodes[node]
                elems = sel_node['mol'].elements_number
                electrons = af.adjust_electrons(sel_node['mol'])
                new_inter = Intermediate(code=node, molecule=sel_node['mol'], graph=sel_node['graph'], energy=sel_node['energy'], entropy=sel_node['entropy'] , electrons=electrons, phase='cat')
                new_inter._graph = new_inter.gen_graph()
                network_dict[key]['intermediates'][node] = new_inter
            except KeyError:
                pass
        
        for edge in list(network.edges()):
            try:
                inters = (network_dict[key]['intermediates'][edge[0]], network_dict[key]['intermediates'][edge[1]])

                if len(inters[0].molecule) > len(inters[1].molecule):
                    comp_1 = (inters[0], surf_inter)
                    comp_2 = (inters[1], h_inter)
                else:
                    comp_1 = (inters[1], surf_inter)
                    comp_2 = (inters[0], h_inter)
                inters = (comp_1, comp_2)
                new_ts = TransitionState(components=inters, r_type='C-H')
                for t_state in network_dict[key]['ts']:
                    if t_state.components == new_ts.components:
                        break
                else:
                    network_dict[key]['ts'].append(new_ts)
            except KeyError:
                pass
    return network_dict

def add_energies_to_dict(network_dict, energ_entr_dict):
    for net in network_dict.values():
        for code, inter in net['intermediates'].items():
            test = [n for n in energ_entr_dict.values()]
            for i in test:
                for vals in i.values():
                    try:
                        inter.energy = vals[code]['e_zpe_solv']
                        inter.entropy = vals[code]['S_meV']
                    except KeyError:
                        pass
    return network_dict

def oh_bond_breaks(ts_list: list) -> None:

    for t_state in ts_list:
        check_bonds = []
        if t_state.r_type == 'C-H':
            comps = t_state.components
            for state in comps:
                for item in list(state):
                    if item.is_surface or item.code == '010101':
                        continue
                    else:
                        oxy = item.molecule['O']
                        if oxy:
                            counter = 0
                            for atom in oxy:
                                counter += len(atom.connections)
                            check_bonds.append(counter)
                        else:
                            check_bonds.append(0)
            if check_bonds[0] != check_bonds[1]:
                t_state.r_type = 'O-H'


def ts_energies(ts_states: list, neb_dict: dict, neb_df, surf_inter) -> None:
    for reaction in ts_states:
        reaction.energy = 0.0
        reaction.entropy = 0.0
        if reaction.r_type == 'C-C':
            fs_components = reaction.components[0]
            is_components = reaction.components[1]

            for molecule in fs_components:
                if molecule.code == '000000':
                    continue
                else:
                    fs_graph = molecule.graph

            is_graphs = []
            for molecule in is_components:
                if molecule.code == '000000':
                    continue
                else:
                    is_graphs.append(molecule.graph)
            possible_rxns = []
            for rxn, ts_data in neb_dict.items():
                try:
                    if nx.is_isomorphic(fs_graph, ts_data['fs']['conn_graph'][0], node_match=lambda x, y: x["elem"] == y["elem"]):
                        fs_system = ts_data['fs']['conn_graph'][0]
                        possible_rxns.append(rxn)
                except TypeError:
                    continue

            for poss_rxn in possible_rxns:
                reaction_db = neb_dict.get(poss_rxn)
            
                dict_graphs = (reaction_db.get('is').get('conn_graph'))
                for comb in it.product(dict_graphs, is_graphs):
                    if nx.is_isomorphic(comb[0], comb[1], node_match=lambda x, y: x["elem"] == y["elem"]):
                        # Getting the energies from neb_db
                        try:
                            
                            zpe_energ = neb_df.loc[neb_df['rxn'] == poss_rxn].e_zpe_solv.values[0]
                            entropy = neb_df.loc[neb_df['rxn'] == poss_rxn].S_meV.values[0]
                            zpe_energ_surf = zpe_energ + surf_inter.energy
                            if np.isnan(zpe_energ):
                                zpe_energ_surf = 0.0
                            if np.isnan(entropy):
                                entropy = 0.0
                            reaction.energy = zpe_energ_surf
                            reaction.entropy = entropy
                        except:
                            continue
                        break

def gen_gas_dict(gas_df):
    gas_dict = {gas_code: {'mol': None, 'conn_graph': None, 'energy': None, 
                       'entropy': None, 'electrons': None} for gas_code in gas_df.gas}


    for gas_code in gas_df.gas:
        gas_data = gas_df.loc[gas_df['gas'] == gas_code]

        gas_mol = pickle.loads(gas_data.contcar_object.values[0])
        gas_dict[gas_code]['mol'] = gas_mol

        gas_graph = pickle.loads(gas_data.conn_graph.values[0])
        gas_dict[gas_code]['conn_graph'] = gas_graph

        gas_dict[gas_code]['energy'] = gas_data.e_zpe.values[0]
        gas_dict[gas_code]['entropy'] = gas_data.S_meV.values[0]

        if np.isnan(gas_dict[gas_code]['energy']):
            gas_dict[gas_code]['energy'] = 0.0
        if np.isnan(gas_dict[gas_code]['entropy']):
            gas_dict[gas_code]['entropy'] = 0.0

        electrons = af.adjust_electrons(gas_mol)
        gas_dict[gas_code]['electrons'] = electrons
    
    return gas_dict

def gen_gas_inter_dict(gas_dict: dict) -> dict:
    gas_inter_dict = {}
    for gas_code, gas_data in gas_dict.items():
        new_inter = Intermediate(gas_code, molecule=gas_data['mol'], energy=gas_data['energy'], 
                                entropy=gas_data['entropy'], electrons=gas_data['electrons'], phase='gas')
        gas_inter_dict[gas_code] = new_inter

    return gas_inter_dict

