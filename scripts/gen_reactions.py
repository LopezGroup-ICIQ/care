import multiprocessing as mp
import argparse
from pickle import dump, load

from care.rnet.gen_rxn_net import generate_reactions

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', type=str, dest='i', help="Path to the folder containing the intermediates.pkl file.")
    argparser.add_argument('-ncores', type=int, dest='ncores', help="Number of cores to use.", default=mp.cpu_count())
    args = argparser.parse_args()
    with open(f"{args.i}/intermediates.pkl", "rb") as infile:
        intermediate_dict = load(infile)
    reactions_list = generate_reactions(intermediate_dict, args.ncores)
    with open(f"{args.i}/reactions.pkl", "wb") as outfile:
        dump(reactions_list, outfile)

    print('Generated {} elementary reactions for the reaction network {} and saved in {}/reactions.pkl'.format(len(reactions_list), args.i, args.i))
    


