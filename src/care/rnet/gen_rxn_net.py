from care.rnet.utilities.additional_funcs import break_and_connect
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.elementary_reaction import ElementaryReaction
from care.rnet.networks.utils import gen_adsorption_reactions, gen_rearrangement_reactions
from ase import Atoms
import multiprocessing as mp


# def generate_rxn_net(slab_ase_obj: Atoms, 
#                      ncc: int,
#                      noc: int) -> ReactionNetwork:
#     """
#     Generate ReactionNetwork given surface and network carbon cutoff (ncc).

#     Parameters
#     ----------
#     slab_ase_obj : Atoms
#         ASE object of the metal surface.
#     ncc : int
#         Network carbon cutoff.
#     noc : int
#         Network oxygen cutoff.

#     Returns
#     -------
#     ReactionNetwork
#         Reaction network.
#     """
    
#     # 1) Generate all the intermediates
#     t0 = time.time()
#     intermediate_dict = generate_intermediates(ncc, noc)    

#     print('Time to generate intermediates: {:.2f} s'.format(time.time()-t0))   
#     surf_inter = Intermediate.from_molecule(slab_ase_obj, code='0000000000*', is_surface=True, phase='surf')

#     # 2) Generate the network dictionary
#     t3 = time.time()
#     print('Generating the network dictionary...')
#     inter_objs_dict = gen_inter_objs(intermediate_dict)
#     print('Time to generate the network dictionary: {:.2f} s'.format(time.time()-t3))

#     # 3) Generate the reaction network
#     t4 = time.time()
#     rxn_net = ReactionNetwork(intermediates={'0000000000': surf_inter}, ncc=ncc)
#     print('Time to generate primitive reaction network: {:.2f} s'.format(time.time()-t4))
#     t5 = time.time()
#     total_len_inter_add = 0
#     for item in intermediate_dict.keys():
#         select_net = inter_objs_dict[item]
#         rxn_net.add_intermediates(select_net['intermediates'])
#         total_len_inter_add += len(select_net['intermediates'])
#     print('Total number of intermediates added to the reaction network: {}'.format(total_len_inter_add))
#     print(len(rxn_net.intermediates))
#     print('Time to add H breaking intermediates and reactions to the reaction network: {:.2f} s'.format(time.time()-t5))
#     t6 = time.time()
#     breaking_reactions = break_and_connect(rxn_net.intermediates)
#     print('Time to generate breaking and connecting reactions: {:.2f} s'.format(time.time()-t6))
#     t7 = time.time()
#     rxn_net.add_reactions(breaking_reactions)
#     print('Time to add breaking reactions: {:.2f} s'.format(time.time()-t7))
#     adsorption_reactions = gen_adsorption_reactions(rxn_net.intermediates, surf_inter)
#     rxn_net.add_reactions(adsorption_reactions)
#     rearrangement_reactions = gen_rearrangement_reactions(intermediate_dict, rxn_net.intermediates)
#     rxn_net.add_reactions(rearrangement_reactions)
#     print('rearrangement_reactions: ', len(rearrangement_reactions))
#     return rxn_net

def generate_reactions(intermediates_dict: dict[str, Intermediate], 
                       ncores: int=mp.cpu_count()) -> list[ElementaryReaction]:
    """
    Generate all the elementary reactions for a given set of surface reaction intermediates.
    Reactions included are bond-breaking/forming steps, adsorption/desorption steps and rearrangement steps.

    Parameters
    ----------
    intermediate_dict : dict[int, Intermediate]
        Dictionary containing the intermediates of the reaction network.
        each key is the intermediate code, while each value is the Intermediate instance.
        Generated with gen_intermediates.py script.
    Returns
    -------
    reactions_list : list[ElementaryReaction]
        List of all the elementary reactions of the reaction network.
    """
    surf_inter = Intermediate.from_molecule(Atoms(), code='0000000000*', is_surface=True, phase='surf')
    intermediates_dict['0000000000*'] = surf_inter  # empty surface Intermediate (the specific surface is defined afterwards)
    reactions_list = []
    bb_steps = break_and_connect(intermediates_dict, ncores)
    reactions_list.extend(bb_steps)   
    print("Bond-breaking steps: {}".format(len(bb_steps))) 
    ads_steps = gen_adsorption_reactions(intermediates_dict)
    reactions_list.extend(ads_steps)
    print("Adsorption steps: {}".format(len(ads_steps)))
    rearr_steps = gen_rearrangement_reactions(intermediates_dict)
    reactions_list.extend(rearr_steps)
    print("Rearrangement steps: {}".format(len(rearr_steps)))
    return reactions_list