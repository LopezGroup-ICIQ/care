"""
Run kinetic simulation for CRN constructed with CARE.
"""

import argparse
import os
from pickle import dump, load
import tomllib

from care import ReactionNetwork

def main():
    """
    Parse .toml configuration file and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="Run microkinetic simulation with CARE."
    )
    PARSER.add_argument(
        "-i",
        type=str,
        dest="input",
        help="Path to .toml configuration file.",
    )
    PARSER.add_argument(
        "-crn",
        type=str,
        dest="crn",
        help="Path to CRN pickle file already evaluated energetically.",
    )
    PARSER.add_argument(
        "-o",
        type=str,
        dest="output",
        help="output file name"
    )

    ARGS = PARSER.parse_args()

    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
        print(f"{LOGO}\n")

    # Output file name
    if ARGS.output is None:
        ARGS.output = f"mkm_C{ARGS.ncc}O{ARGS.noc}_cyclic{ARGS.cyclic}_rearr{ARGS.rearr}_electro{ARGS.electro}"

    # Load CRN blueprint
    with open(ARGS.crn, "rb") as f:
        crn = load(f)

    if not isinstance(crn, ReactionNetwork):
        raise TypeError("The input CRN file does not contain a valid ReactionNetwork object.")

    # Load evaluation settings
    with open(ARGS.input, "rb") as f:
        config = tomllib.load(f)

    if "mkm" in config.keys() and "operating_conditions" in config.keys() and "initial_conditions" in config.keys():
        MKM_SWITCH = True
    else:
        raise KeyError("Running kinetic simulations requires the fields 'mkm', 'operating_conditions', and 'initial_conditions' in the input .toml file.")

    PH = config["operating_conditions"]["pH"] if crn.type == "electro" else None
    U = config["operating_conditions"]["U"] if crn.type == "electro" else None
    T = config["operating_conditions"]["temperature"]
    P = config["operating_conditions"]["pressure"]


    print(
        f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━ Run kinetic simulation  ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    )

    y = crn.run_microkinetic(
            iv=config["initial_conditions"],
            oc={"T": T, "P": P, "U": U, "pH": PH},
            **config["mkm"]
        )

    print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Kinetic simulation ended ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

    print(f"Total time: {y['time']:.2f} s")
    with open(ARGS.output+'.pkl', "wb") as f:
        dump(y, f)
        print(f"MKM results saved to {ARGS.output+'.pkl'}")

if __name__ == '__main__':
    main()
