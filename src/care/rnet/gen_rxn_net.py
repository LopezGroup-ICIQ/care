from care.rnet.utilities.additional_funcs import break_and_connect
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.reaction_network import ReactionNetwork
from care.rnet.networks.utils import generate_dict, generate_network_dict, gen_adsorption_reactions, gen_rearrangement_reactions
from care.rnet.gen_intermediates import generate_intermediates
import networkx as nx
from ase import Atoms
import time


def generate_rxn_net(slab_ase_obj: Atoms, 
                     ncc: int,
                     noc: int) -> ReactionNetwork:
    """
    Generate ReactionNetwork given surface and network carbon cutoff (ncc).

    Parameters
    ----------
    slab_ase_obj : Atoms
        ASE object of the metal surface.
    ncc : int
        Network carbon cutoff.
    noc : int
        Network oxygen cutoff.

    Returns
    -------
    ReactionNetwork
        Reaction network.
    """
    
    # 1) Generate all the intermediates
    t0 = time.time()
    intermediate_dict, map_dict = generate_intermediates(ncc, noc)
    print('Time to generate intermediates: {:.2f} s'.format(time.time()-t0))   
    surf_inter = Intermediate.from_molecule(slab_ase_obj, code='0000000000*', is_surface=True, phase='surf')

    # 2) Generate the network dictionary
    t1 = time.time()
    inter_att = generate_dict(intermediate_dict)
    print('Time to generate the dictionary: {:.2f} s'.format(time.time()-t1))
    t2 = time.time()
    for network, graph in map_dict.items():
        nx.set_node_attributes(graph, inter_att[network])
    print('Time to set node attributes: {:.2f} s'.format(time.time()-t2))
    t3 = time.time()
    print('Generating the network dictionary...')
    network_dict = generate_network_dict(map_dict, surf_inter)
    int_net = [intermediate for intermediate in intermediate_dict.keys()]
    print('Time to generate the network dictionary: {:.2f} s'.format(time.time()-t3))

    # 3) Generate the reaction network
    t4 = time.time()
    rxn_net = ReactionNetwork(intermediates={'0000000000': surf_inter}, ncc=ncc)
    print('Time to generate primitive reaction network: {:.2f} s'.format(time.time()-t4))
    t5 = time.time()
    for item in int_net:
        select_net = network_dict[item]
        rxn_net.add_intermediates(select_net['intermediates'])
        rxn_net.add_reactions(select_net['reactions'])
    print('Time to add H breaking intermediates and reactions to the reaction network: {:.2f} s'.format(time.time()-t5))
    t6 = time.time()
    breaking_reactions = break_and_connect(rxn_net.intermediates)
    print('Time to generate breaking and connecting reactions: {:.2f} s'.format(time.time()-t6))
    t7 = time.time()
    rxn_net.add_reactions(breaking_reactions)
    print('Time to add breaking reactions: {:.2f} s'.format(time.time()-t7))
    gas_molecules, desorption_reactions = gen_adsorption_reactions(rxn_net.intermediates, surf_inter)
    rxn_net.add_intermediates(gas_molecules)
    rxn_net.add_reactions(desorption_reactions)
    rearrangement_reactions = gen_rearrangement_reactions(intermediate_dict, rxn_net.intermediates)
    rxn_net.add_reactions(rearrangement_reactions)
    return rxn_net