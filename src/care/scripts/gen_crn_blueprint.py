"""
Generate chemical reaction network blueprint with CARE.
"""

import argparse
import os
from pathlib import Path
from time import time

from care import ReactionNetwork
from care.io import save_network


def main():
    """
    Parse configuration arguments and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="Generate chemical reaction network blueprint with CARE."
    )
    
    # Use '--' for long arguments. Grouping logically.
    PARSER.add_argument(
        "--reactants",
        type=str,
        nargs="+",
        help="SMILES of the reactants of the network.",
    )
    PARSER.add_argument(
        "--products",
        type=str,
        nargs="+",
        help="SMILES of the products of the network.",
    )
    PARSER.add_argument(
        "--ncc",
        type=int,
        help="Network Carbon Cutoff (i.e., max number of C atoms in the intermediates).",
    )
    PARSER.add_argument(
        "--noc",
        type=int,
        help="Network Oxygen Cutoff (i.e., max number of O atoms in the intermediates).",
    )
    PARSER.add_argument(
        "--cs",
        type=str,
        nargs="+",
        help="List of SMILES of the molecules from which the CRN is constructed.",
    )
    PARSER.add_argument(
        "--cyclic",
        action="store_true",
        help="Include cyclic species (ncc>=2). Default is False.",
    )
    PARSER.add_argument(
        "--rearr",
        action="store_true",
        help="Include [1,2]-H shift rearrangement steps. Default is False.",
    )
    PARSER.add_argument(
        "--electro",
        action="store_true",
        help="Build blueprint in electrochemical conditions. Default is False.",
    )
    
    PARSER.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file name (e.g., network.json).",
    )
    PARSER.add_argument(
        "--ncpu",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU cores to use. Default is all available cores.",
    )

    ARGS = PARSER.parse_args()

    # Robust path resolution using pathlib
    logo_path = Path(__file__).resolve().parent.parent / "logo.txt"
    try:
        print(f"{logo_path.read_text()}\n")
    except FileNotFoundError:
        pass            

    t0 = time()
    print(f"\n┏━━━━━━━━━━━━━━━━━━ Generating the CRN blueprint ━━━━━━━━━━━━━━━━━━┓\n")
    if ARGS.reactants and ARGS.products:
        print(f"Reactants SMILES: {', '.join(ARGS.reactants)}")
        print(f"Products SMILES:  {', '.join(ARGS.products)}")
        crn = ReactionNetwork.from_species(
            reactants=ARGS.reactants, 
            products=ARGS.products, 
            cyclic=ARGS.cyclic, 
            additional_rxns=ARGS.rearr, 
            electro=ARGS.electro, 
            num_cpu=ARGS.ncpu,
            show_progress=True
        )
    elif ARGS.ncc is not None and ARGS.noc is not None:
        print(f"ncc={ARGS.ncc}, noc={ARGS.noc}")
        crn = ReactionNetwork.from_cutoffs(
            ncc=ARGS.ncc, 
            noc=ARGS.noc, 
            cyclic=ARGS.cyclic, 
            additional_rxns=ARGS.rearr, 
            electro=ARGS.electro, 
            num_cpu=ARGS.ncpu,
            show_progress=True
        )
    elif ARGS.cs:
        print(f"Input chemical space (SMILES): {', '.join(ARGS.cs)}")
        crn = ReactionNetwork.from_chemical_space(
            cs=ARGS.cs,
            additional_rxns=ARGS.rearr, 
            electro=ARGS.electro, 
            cyclic=ARGS.cyclic, 
            num_cpu=ARGS.ncpu,
            show_progress=True
        )
    else:
        PARSER.error("You must provide either --reactants/--products, --ncc/--noc, or --cs to generate a network.")

    t = time() - t0
    print("\n┗━━━━━━━━━━━━━━━━━━━━━━ CRN blueprint generated ━━━━━━━━━━━━━━━━━━━┛\n")
    print(f"Total time: {t:.2f} s")
    out_file = ARGS.output if ARGS.output.endswith('.json') else f"{ARGS.output}.json"
    save_network(crn, out_file)
    print(f"Saved to: {out_file}")


if __name__ == '__main__':
    main()