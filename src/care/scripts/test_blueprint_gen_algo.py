"""
Script to test the blueprint generation algorithm
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from care import gen_blueprint

def test_blueprint_gen():
    """
    Parse .toml configuration file and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="Test CRN blueprint generation algorithm in terms of elapsed time as function of ncc, noc, etc."
    )
    PARSER.add_argument(
        "-max_ncc",
        type=int,
        dest="max_ncc",
        default=2,
        help="Max Network Carbon Cutoff (i.e., max number of C atoms in the intermediates). Default is 2.",
    )
    PARSER.add_argument(
        "-max_noc",
        type=int,
        dest="max_noc",
        default=6,
        help="Max Network Oxygen Cutoff (i.e., max number of O atoms in the intermediates). Default is 6.",
    )
    PARSER.add_argument(
        "-cyclic",
        type=bool,
        dest="cyclic",
        default=False,
        help="Include cyclic species (ncc>=2). Default is False.",
    )
    PARSER.add_argument(
        "-rearr",
        type=bool,
        dest="rearr",
        default=False,
        help="Include [1,2]-H shift rearrengement steps. Default is False.",
    )
    PARSER.add_argument(
        "-electro",
        type=bool,
        dest="electro",
        default=False,
        help="Build blueprint in electrochemical conditions. Default is False.",
    )
    PARSER.add_argument(
        "-o",
        type=str,
        dest="o",
        help="output file name",
        default="test_blueprint_gen.csv"
    )
    PARSER.add_argument(
        "-max_ncpu",
        type=int,
        dest="max_cores",
        help="Max Number of CPU cores to use for the CRN blueprint generation. Default is the number of CPU cores available.",
        default=os.cpu_count(),
    )

    args = PARSER.parse_args()

    NCC_MAX = args.max_ncc
    NOC_MAX = args.max_noc
    CPUS_MAX = args.max_cores
    CYCLIC = args.cyclic
    REARR = args.rearr
    ELECTRO = args.electro

    info_dict = {}

    for c in range(1, NCC_MAX+1):
        for o in range(1, min(2*c+3, NOC_MAX+1)):
            for ncpu in range(1, CPUS_MAX+1):
                bp_args = (c, o, ncpu)
                print(bp_args)
                t0 = time.time()
                inters, steps = gen_blueprint(c, o, CYCLIC, REARR, ELECTRO, num_cpu=ncpu)
                tf_sec = time.time() - t0
                ns, nr = len(inters), len(steps)
                sum_inters = 0
                for step in steps:
                    sum_inters += len(step.stoic)
                sparsity = (1 - sum_inters / ((ns+1)*nr)) * 100.0
                info_dict[bp_args] = (ns, nr, sparsity, tf_sec)

    ncc, noc, add_rxn, ns, nr, sparsity, time_s, ncpus = [], [], [], [], [], [], [], []
    cyclic, electro = [], []
    for key, value in info_dict.items():
        ncc.append(key[0])
        noc.append(key[1])
        ns.append(value[0])
        nr.append(value[1])
        sparsity.append(value[2])
        time_s.append(value[3])
        ncpus.append(key[2])
        cyclic.append(CYCLIC)
        electro.append(ELECTRO)
        add_rxn.append(REARR)

    df = pd.DataFrame({'ncc': ncc,
                       'noc': noc,
                       'rearr': add_rxn,
                       'electro': electro,
                       'cyclic': cyclic,
                       'ns': ns,
                       'nr': nr,
                       'sparsity': sparsity,
                       'time_s': time_s,
                       'ncpus': ncpus})
    df['log_ns'] = np.log10(df['ns'])
    df['log_nr'] = np.log10(df['nr'])
    df['core_s'] = df['time_s'] * df['ncpus']
    df.to_csv(args.o, index=False)

if __name__ == "__main__":
    test_blueprint_gen()
