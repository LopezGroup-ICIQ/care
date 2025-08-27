"""
Evaluate chemical reaction network blueprint with CARE.
"""

import argparse
import os
from time import time, sleep
import tomllib
from pickle import dump, dumps, load

import dask
from dask.distributed import Client, LocalCluster

from care import ReactionNetwork, load_surface
from care.crn.utils.electro import Electron
from care.evaluators import load_inter_evaluator, load_reaction_evaluator
from care.scripts import setup_logging, load_x, predict


def main():
    """
    Parse .toml configuration file and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="Evaluate species and reaction properties of a chemical reaction network blueprint with CARE."
    )
    PARSER.add_argument(
        "-i",
        type=str,
        dest="input",
        help="Path to .toml configuration file.",
    )
    PARSER.add_argument(
        "-bp",
        type=str,
        dest="bp",
        help="Path to CRN blueprint file.",
    )
    PARSER.add_argument(
        "-o",
        type=str,
        dest="output",
        help="output file name.",
        default="crn"
    )
    PARSER.add_argument(
        "-ncpu",
        type=int,
        dest="num_cpu",
        help="Number of CPU cores to use for parallelizing intermediate evaluation. Default is the number of CPU cores available.",
        default=os.cpu_count(),
    )
    PARSER.add_argument(
        '--log', 
        type=str, 
        help='Path to run log file', 
        default="care.log"
    )

    ARGS = PARSER.parse_args()
    setup_logging(ARGS.log)

    # Load CRN blueprint
    with open(ARGS.bp, "rb") as f:
        inters, rxns = load(f)

    # Load evaluation settings
    with open(ARGS.input, "rb") as f:
        config = tomllib.load(f)

    # Check on input toml entries
    if "surface" not in config.keys():
        raise KeyError("'surface' field definition not found in input .toml file. Please define the surface where you want to evaluate your CRN.")
    if "evaluator" not in config.keys():
        raise KeyError("'evaluator' field definition not found in the input .toml file. Please define the energy evaluator.")

    surface = load_surface(**config["surface"])

    model_name = config["evaluator"]["model"]
    del config["evaluator"]["model"]
    inter_evaluator = load_inter_evaluator(model_name, surface, **config["evaluator"])
    rxn_evaluator = load_reaction_evaluator(model_name, inter_evaluator, **config["evaluator"])
    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
        print(f"{LOGO}\n")

    # 2. Evaluation of the adsorbed intermediates in the CRN with GAME-Net-UQ
    print(
        f"\n┏━━━━━━━━━━━━ Evaluating CRN ━━━━━━━━━━━┓\n"
    )
    t0 = time()
    # INTERMEDIATE EVALUATION
    print(f"Energy estimation of the {len(inters)} intermediates...")
    print("Intermediates energy calculator: ", inter_evaluator)
    cluster = LocalCluster(n_workers=ARGS.num_cpu,
                           threads_per_worker=1, 
                           ip='127.0.0.1', 
                           scheduler_port=0, 
                           dashboard_address=":0")
    client = Client(address=cluster)
    print(f"Dask dashboard available at: {cluster.dashboard_link}")
    tasks = [load_x(intermediate) for intermediate in inters.values()]
    dask.utils.format_bytes(len(dumps(inter_evaluator)))
    dmodel = dask.delayed(inter_evaluator)
    predictions = [predict(task, dmodel) for task in tasks]
    predictions = dask.compute(*predictions)
    intermediates = {inter.code: inter for inter in predictions}
    for rxn in rxns:
        rxn.update_intermediates(intermediates)
    ti = time()
    print(f"Total intermediate evaluation time: {ti - t0:.2f} s")

    # REACTION EVALUATION
    print(f"\nEnergy estimation of the {len(rxns)} reactions...")
    print("Reaction properties calculator: ", rxn_evaluator)
    tasks = [load_x(reaction) for reaction in rxns]
    dask.utils.format_bytes(len(dumps(rxn_evaluator)))
    dmodel = dask.delayed(rxn_evaluator)
    predictions = [predict(task, dmodel) for task in tasks]
    predictions = dask.compute(*predictions)
    client.shutdown()  #retire_workers()
    client.close()
    cluster.close()
    sleep(1)
    rxns = sorted(predictions)
    tr = time()
    print(f"Total reaction evaluation time: {tr - ti:.2f} s")

    print(
                "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
            )

    for r in rxns:
        if Electron in r.reactants or Electron in r.products:
            crn_type = "electrochemical"
            break
    else:
        crn_type = "thermal"

    crn = ReactionNetwork(
            intermediates=intermediates,
            reactions=rxns,
            surface=surface,
            ncc=max([i['C'] for i in inters.values()]),
            noc=max([i['O'] for i in inters.values()]),
            type=crn_type,
        )

    print(f"Total time: {(time() - t0):.2f} s")
    # Save the blueprint
    with open(ARGS.output+'.pkl', "wb") as f:
        dump(crn, f)
        print(f"CRN saved to {ARGS.output+'.pkl'}")

if __name__ == '__main__':
    main()
