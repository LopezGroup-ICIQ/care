import os
from GAMERNet import H_PATH
from GAMERNet.rnet.utilities.additional_funcs import break_and_connect
from GAMERNet.rnet.networks.networks import Intermediate, OrganicNetwork
from GAMERNet.rnet.networks.network_functions import generate_dict, generate_network_dict, oh_bond_breaks
from pyRDTP.geomio import MolObj, ASEAtoms, file_to_mol
import networkx as nx

def organic_network(slab_ase_obj, intermediate_dict, map_dict):
    # Converting the slab_ase_obj to pyrdtp
    uni_coords = ASEAtoms(slab_ase_obj).universal_convert()
    slab_mol_obj = MolObj()
    slab_mol_obj.universal_read(uni_coords)
    slab_mol_obj = slab_mol_obj.write()

    hydro_path = H_PATH
    hydrogen = file_to_mol(hydro_path, 'contcar')['H']

    surf_inter = Intermediate.from_molecule(slab_mol_obj, code='000000', energy=0.0, entropy=0, is_surface=True, phase='surface')
    surf_inter.electrons = 0
    h_inter = Intermediate.from_molecule(hydrogen, code='010101', energy=0.0, entropy=0.0, phase='cat')
    h_inter.electrons = 1

    inter_att = generate_dict(intermediate_dict)
    for network, graph in map_dict.items():
        nx.set_node_attributes(graph, inter_att[network])

    network_dict = generate_network_dict(map_dict, surf_inter, h_inter)

    int_net = [intermediate for intermediate in intermediate_dict.keys()]

    rxn_net = OrganicNetwork()
    for item in int_net:
        select_net = network_dict[item]
        rxn_net.add_intermediates(select_net['intermediates'])
        rxn_net.add_intermediates({'000000': surf_inter, '010101': h_inter})
        rxn_net.add_ts(select_net['ts'])

    for inter in rxn_net.intermediates.values():
            inter.molecule.connection_clear()
            inter.molecule.connectivity_search_voronoi()

    oh_bond_breaks(rxn_net.t_states)

    breaking_ts = break_and_connect(rxn_net, surface=surf_inter)

    # Adding the TSs to the network
    rxn_net.add_ts(breaking_ts) 

    return rxn_net