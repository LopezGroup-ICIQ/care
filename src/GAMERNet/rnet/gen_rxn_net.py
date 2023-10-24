from GAMERNet.rnet.utilities.additional_funcs import break_and_connect
from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.reaction_network import ReactionNetwork
from GAMERNet.rnet.networks.utils import generate_dict, generate_network_dict, gen_adsorption_reactions
from GAMERNet.rnet.gen_intermediates import generate_intermediates
import networkx as nx
from ase import Atoms

def generate_rxn_net(slab_ase_obj: Atoms, 
                     ncc: int) -> ReactionNetwork:
    """
    Generate ReactionNetwork given surface and network carbon cutoff (ncc).

    Parameters
    ----------
    slab_ase_obj : Atoms
        ASE object of the metal surface.
    ncc : int
        Network carbon cutoff.

    Returns
    -------
    ReactionNetwork
        Reaction network.
    """
    
    # 1) Generate all the intermediates
    intermediate_dict, map_dict = generate_intermediates(ncc)    
    surf_inter = Intermediate.from_molecule(slab_ase_obj, code='00000*', is_surface=True, phase='surf')

    # 2) Generate the network dictionary
    inter_att = generate_dict(intermediate_dict)
    for network, graph in map_dict.items():
        nx.set_node_attributes(graph, inter_att[network])
    network_dict = generate_network_dict(map_dict, surf_inter)
    int_net = [intermediate for intermediate in intermediate_dict.keys()]

    # 3) Generate the reaction network
    rxn_net = ReactionNetwork(intermediates={'00000': surf_inter}, ncc=ncc)
    for item in int_net:
        select_net = network_dict[item]
        rxn_net.add_intermediates(select_net['intermediates'])
        rxn_net.add_reactions(select_net['reactions'])
    breaking_reactions = break_and_connect(rxn_net.intermediates, surf_inter)
    rxn_net.add_reactions(breaking_reactions)
    gas_molecules, desorption_reactions = gen_adsorption_reactions(rxn_net.intermediates, surf_inter)
    rxn_net.add_intermediates(gas_molecules)
    rxn_net.add_reactions(desorption_reactions)
    return rxn_net