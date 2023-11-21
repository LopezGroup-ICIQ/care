import argparse
from pickle import dump
import os
import multiprocessing as mp

from care.rnet.netgen_fns import generate_inters_and_rxns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate intermediates of the reaction network.')
    parser.add_argument('-ncc', type=int, help='Network Carbon Cutoff, i.e., max number of C atoms in the intermediates.', dest='ncc')
    parser.add_argument('-noc', type=int, help='Network Oxygen Cutoff, i.e., max number of O atoms in the intermediates.', dest='noc', default=-1)
    parser.add_argument('-ncores', type=int, help='Number of cores to use.', dest='ncores', default=mp.cpu_count())
    args = parser.parse_args()

    if args.noc == -1:
        noc = args.ncc*2 +2

    output_dir = f'C{args.ncc}_O{noc}'
    os.makedirs(output_dir, exist_ok=True)

    intermediates, reactions = generate_inters_and_rxns(args.ncc, args.noc, ncores=args.ncores)
    with open(f'{output_dir}/intermediates.pkl', 'wb') as f:
        dump(intermediates, f)

    with open(f'{output_dir}/reactions.pkl', 'wb') as f:
        dump(reactions, f)

    print(f'Generated {len(intermediates)} intermediates of the C{args.ncc}O{noc} reaction network and saved in {output_dir}/intermediates.pkl')
    print(f'Generated {len(reactions)} reactions of the C{args.ncc}O{noc} reaction network and saved in {output_dir}/reactions.pkl')
    