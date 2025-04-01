"""
Evaluate chemical reaction network blueprint with CARE.
"""

import argparse
import os
from time import time
import tomllib
from pickle import dump, dumps, load
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


import dask
from dask.distributed import Client, LocalCluster
from rich.progress import Progress

from care import ReactionNetwork
from care.crn.utils.electro import Electron
from care.evaluators import load_surface, load_inter_evaluator, load_reaction_evaluator

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
        help="output file name."
    )
    PARSER.add_argument(
        "-ncpu",
        type=int,
        dest="num_cpu",
        help="Number of CPU cores to use for parallelizing intermediate evaluation. Default is the number of CPU cores available.",
        default=os.cpu_count(),
    )

    ARGS = PARSER.parse_args()

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
    print(" Energy estimation of the intermediates...")
    print(" Intermediates energy calculator: ", inter_evaluator)    
    cluster = LocalCluster(n_workers=ARGS.num_cpu, 
                           threads_per_worker=1)
    client = Client(address=cluster)
    print(client.dashboard_link)
    @dask.delayed
    def load_inter(inter):
        return inter
    tasks = [load_inter(intermediate) for intermediate in inters.values()]
    @dask.delayed
    def predict(inter, dmodel):
        dmodel(inter)
        return inter
    dask.utils.format_bytes(len(dumps(inter_evaluator)))
    dmodel = dask.delayed(inter_evaluator)
    predictions = [predict(task, dmodel) for task in tasks]
    predictions = dask.compute(*predictions)
    intermediates = {inter.code: inter for inter in predictions}

    # REACTION EVALUATION
    print("\n Energy estimation of the reactions...")
    rxn_evaluator = load_reaction_evaluator(model_name, intermediates, **config["evaluator"])
    print(" Reaction properties calculator: ", rxn_evaluator)
    with Progress() as progress:
        task = progress.add_task(" [green]Processing...", total=len(rxns))
        processed_items = 0
        for reaction in rxns:
            rxn_evaluator.eval(reaction)
            processed_items += 1
            progress.update(
                task,
                advance=1,
                description=f" [green]Processing {processed_items}/{len(rxns)}...",
            )
    rxns = sorted(rxns)
    t = time() - t0

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

    print(f"Total time: {t:.2f} s")
    # Save the blueprint
    with open(ARGS.output+'.pkl', "wb") as f:
        dump(crn, f)
        print(f"CRN saved to {ARGS.output+'.pkl'}")

if __name__ == '__main__':
    main()
