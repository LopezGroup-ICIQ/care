from care.rnet.utilities.additional_funcs import break_and_connect
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.elementary_reaction import ElementaryReaction
from care.rnet.networks.utils import gen_adsorption_reactions, gen_rearrangement_reactions
from ase import Atoms
import argparse
from pickle import dump, load


def generate_reactions(intermediates_dict: dict[str, Intermediate]) -> list[ElementaryReaction]:
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
    bb_steps = break_and_connect(intermediates_dict)
    reactions_list.extend(bb_steps)   
    print("Bond-breaking steps: {}".format(len(bb_steps))) 
    ads_steps = gen_adsorption_reactions(intermediates_dict)
    reactions_list.extend(ads_steps)
    print("Adsorption steps: {}".format(len(ads_steps)))
    rearr_steps = gen_rearrangement_reactions(intermediates_dict)
    reactions_list.extend(rearr_steps)
    print("Rearrangement steps: {}".format(len(rearr_steps)))
    return reactions_list

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', type=str, dest='i', help="path to the folder containing the intermediates.pkl file")
    args = argparser.parse_args()
    with open(f"{args.i}/intermediates.pkl", "rb") as infile:
        intermediate_dict = load(infile)
    reactions_list = generate_reactions(intermediate_dict)
    with open(f"{args.i}/reactions.pkl", "wb") as outfile:
        dump(reactions_list, outfile)

    print('Generated {} elementary reactions for the reaction network {} and saved in {}/reactions.pkl'.format(len(reactions_list), args.i, args.i))
    


