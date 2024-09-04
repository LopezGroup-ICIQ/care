"""
Evaluate chemical reaction network blueprint with CARE.
"""

import argparse
import os
from pickle import dump, load
from time import time
import tomllib

import argparse
import os
import tomllib
import multiprocessing as mp
from pickle import dump, load, dumps
import resource
from rich.progress import Progress
from prettytable import PrettyTable
import cpuinfo
import psutil
import tempfile
from time import time

from rich.progress import Progress

from care import ReactionNetwork, Intermediate
from care.crn.utils.electro import Electron
from care.evaluators import load_surface, load_inter_evaluator, load_reaction_evaluator

lock = mp.Lock()

def evaluate_intermediate(
    chunk_intermediate: list[Intermediate], model, progress_queue, f_path
):
    """
    Evaluates a chunk of intermediates and saves the results to a file.
    For memory efficiency, the results are saved in a file instead of a list.

    Args:
        intermediate (Intermediate): The intermediate.
        model (GameNetUQ): The model.
    """

    eval_inter = {}
    for intermediate in chunk_intermediate:
        model.eval(intermediate)
        eval_inter[intermediate.code] = intermediate

    progress_queue.put(1)

    with lock:
        with open(f_path, "ab") as file:
            pkl_str = dumps(eval_inter)
            file.write(pkl_str)


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

    metal = config["surface"]["metal"]
    hkl = config["surface"]["hkl"]
    surface = load_surface(metal, hkl)

    # Set evaluators
    model_name = config["eval"]["model"]

    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
        print(f"{LOGO}\n")

    # 2. Evaluation of the adsorbed intermediates in the CRN with GAME-Net-UQ
    print(
        f"\n┏━━━━━━━━━━━━ Evaluating CRN on {metal}({hkl}) ━━━━━━━━━━━┓\n"
    )
    t0 = time()
    # INTERMEDIATE EVALUATION
    print(" Energy estimation of the intermediates...")
    inter_evaluator = load_inter_evaluator(model_name, surface, **config["intermediate_args"])
    print(" Intermediates energy calculator: ", inter_evaluator)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    manager = mp.Manager()
    progress_queue = manager.Queue()

    if len(inters) < 10000:
        chunk_size = 1
        tasks = [[intermediate] for intermediate in inters.values()]
    else:
        chunk_size = len(inters) // (ARGS.num_cpu * 10)
        tasks = [
            list(inters.values())[i : i + chunk_size]
            for i in range(0, len(inters), chunk_size)
        ]

    # Create empty folder to store tmp results
    tmp_folder = tempfile.mkdtemp()
    _, tmp_file = tempfile.mkstemp(suffix=".pkl", dir=tmp_folder)

    with Progress() as progress:
        task = progress.add_task(" [green]Processing...", total=None)
        processed_items = 0

        with mp.Pool(ARGS.num_cpu) as pool:
            pool.starmap(
                evaluate_intermediate,
                [
                    (task, inter_evaluator, progress_queue, tmp_file)
                    for task in tasks
                ],
            )

            while not progress_queue.empty():
                progress.update(task, advance=progress_queue.get())
                processed_items += 1

    intermediates = {}
    with open(tmp_file, "rb") as file:
        while True:
            try:
                intermediates.update(load(file))
            except EOFError:
                break

    # REACTION EVALUATION
    print("\n Energy estimation of the reactions...")
    rxn_evaluator = load_reaction_evaluator(model_name, intermediates, **config["reaction_args"])
    print(" Reactions energy calculator: ", rxn_evaluator)
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
