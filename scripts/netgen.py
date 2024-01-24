import argparse
from pickle import dump
import os
import multiprocessing as mp

from care.crn.utilities.build_netgen import gen_chemical_space

def main():
    parser = argparse.ArgumentParser(description='Generate intermediates of the reaction network.')
    parser.add_argument('-ncc', type=int, help='Network Carbon Cutoff, i.e., max number of C atoms in the intermediates.', dest='ncc')
    parser.add_argument('-noc', type=int, help='Network Oxygen Cutoff, i.e., max number of O atoms in the intermediates.', dest='noc', default=-1)
    parser.add_argument('-ncores', type=int, help='Number of cores to use.', dest='ncores', default=mp.cpu_count())
    args = parser.parse_args()

    # If args.noc is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    noc = args.noc if args.noc > 0 else args.ncc*2 + 2

    output_dir = f'C{args.ncc}O{noc}'
    os.makedirs(output_dir, exist_ok=True)

    intermediates, reactions = gen_chemical_space(args.ncc, args.noc, ncores=args.ncores)
    with open(f'{output_dir}/intermediates.pkl', 'wb') as f:
        dump(intermediates, f)

    with open(f'{output_dir}/reactions.pkl', 'wb') as f:
        dump(reactions, f)
    
if __name__ == "__main__":
    main()