from GAMERNet.rnet.utilities.additional_funcs import break_and_connect
from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction
from GAMERNet.rnet.networks.reaction_network import ReactionNetwork
from GAMERNet.rnet.networks.utils import generate_dict, generate_network_dict, classify_oh_bond_breaks, gen_desorption_reactions, gen_associative_desorption_reactions
from GAMERNet.rnet.utilities.functions import get_voronoi_neighbourlist, MolPack
import networkx as nx
from ase import Atoms

def generate_rxn_net(slab_ase_obj: Atoms, 
                     intermediate_dict: dict[int, list], 
                     rxn_dict: dict[int, nx.DiGraph]) -> ReactionNetwork:
    """
    Generates the entire reaction network from the intermediate dictionary and the map dictionary.

    Parameters
    ----------
    slab_ase_obj : Atoms
        ASE object of the metal surface.
    intermediate_dict : dict[int, list]
        Dictionary of the intermediates.
    map_dict : dict[int, nx.DiGraph]
        Dictionary of the reaction networks.

    Returns
    -------
    ReactionNetwork
        Reaction network.
    """
    
    hydrogen_atom_ase = Atoms('H', positions=[[0, 0, 0]])
    surf_inter = Intermediate.from_molecule(slab_ase_obj, code='000000*', energy=0.0, entropy=0.0, is_surface=True, phase='surf')
    surf_inter.electrons = 0
    h_inter = Intermediate.from_molecule(hydrogen_atom_ase, code='010101*', energy=0.0, entropy=0.0, phase='ads')
    h_inter.electrons = 1

    inter_att = generate_dict(intermediate_dict)
    for network, graph in rxn_dict.items():
        nx.set_node_attributes(graph, inter_att[network])

    network_dict = generate_network_dict(rxn_dict, surf_inter, h_inter)
    int_net = [intermediate for intermediate in intermediate_dict.keys()]

    rxn_net = ReactionNetwork()
    for item in int_net:
        select_net = network_dict[item]
        rxn_net.add_intermediates(select_net['intermediates'])
        rxn_net.add_intermediates({'000000': surf_inter})#, '010101': h_inter})
        rxn_net.add_ts(select_net['ts'])

    for inter in rxn_net.intermediates.values():
            if 'conn_pairs' in list(inter.molecule.arrays.keys()):
                del inter.molecule.arrays['conn_pairs']
            inter.molecule.arrays['conn_pairs'] = get_voronoi_neighbourlist(inter.molecule, 0.25, 1, ['C', 'H', 'O'])

    classify_oh_bond_breaks(rxn_net.reactions)
    breaking_ts = break_and_connect(rxn_net, surface=surf_inter)
    rxn_net.add_ts(breaking_ts)
    des_reactions = gen_desorption_reactions(rxn_net, surf_inter)
    rxn_net.add_ts(des_reactions)
    ass_des_reactions = gen_associative_desorption_reactions(rxn_net, surf_inter)
    rxn_net.add_ts(ass_des_reactions)


    for t_state in rxn_net.reactions:
         t_state.energy, t_state.entropy = 0.0, 0.0

    return rxn_net