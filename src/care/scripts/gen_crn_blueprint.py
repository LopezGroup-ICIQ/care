"""
Generate chemical reaction network blueprint with CARE.
"""

import argparse
import os
from pickle import dump
from time import time

from care import gen_blueprint


def main():
    """
    Parse .toml configuration file and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="Generate chemical reaction network blueprint with CARE."
    )
    PARSER.add_argument(
        "-ncc",
        type=int,
        dest="ncc",
        default=None,
        help="Network Carbon Cutoff (i.e., max number of C atoms in the intermediates). Default is 1.",
    )
    PARSER.add_argument(
        "-noc",
        type=int,
        dest="noc",
        default=None,
        help="Network Oxygen Cutoff (i.e., max number of O atoms in the intermediates). Default is 1.",
    )
    PARSER.add_argument(
        "-cs",
        type=str,
        nargs="+",
        dest="cs",
        default=None,
        help="List of SMILES of the molecules from which the CRN is constructed. You can provide cs or ncc and noc. If both are provided, cs is used.",
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
        dest="output",
        help="output file name"
    )
    PARSER.add_argument(
        "-ncpu",
        type=int,
        dest="num_cpu",
        help="Number of CPU cores to use for the CRN generation. Default is the number of CPU cores available.",
        default=os.cpu_count(),
    )

    ARGS = PARSER.parse_args()

    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
        print(f"{LOGO}\n")

    if ARGS.ncc and not ARGS.cs:
        print(
            f"\n┏━━━━━━━━━━━━━━━━━━━━━━━ Generating the CRN(ncc={ARGS.ncc},ncc={ARGS.noc}) blueprint  ━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        )
    else:
        print(
            f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Generating the CRN blueprint  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        )
        print("Input chemical space (SMILES): {}".format(", ".join(ARGS.cs)))
    t0 = time()
    inters, rxns = gen_blueprint(
        ncc=ARGS.ncc,
        noc=ARGS.noc,
        cs=ARGS.cs,
        cyclic=ARGS.cyclic,
        additional_rxns=ARGS.rearr,
        electro=ARGS.electro,
        num_cpu=ARGS.num_cpu,
        show_progress=True
    )
    t = time() - t0
    print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CRN blueprint generated ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

    print(f"Total time: {t:.2f} s")
    # Save the blueprint
    with open(ARGS.output+'.pkl', "wb") as f:
        dump((inters, rxns), f)
        print(f"CRN blueprint saved to {ARGS.output+'.pkl'}")

if __name__ == '__main__':
    main()
