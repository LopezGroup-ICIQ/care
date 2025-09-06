"""
Evaluate chemical reaction network blueprint with CARE.
"""

import argparse
import gc
import os
from time import time
import tomllib
from pickle import dump, load
from tqdm import tqdm

import dask
from dask.distributed import Client, LocalCluster

from care import ReactionNetwork, load_surface
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
        "-bs_rxn",
        type=int,
        dest="batch_size_rxn",
        help="Batch size for reaction evaluation. Default is 256.",
        default=512,
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
    if os.path.exists(ARGS.output + "_intermediates.pkl"):
        with open(ARGS.output + "_intermediates.pkl", "rb") as f:
            print("Loading intermediates from disk...")
            intermediates = load(f)
    else:
        tasks = [load_x(intermediate) for intermediate in inters.values()]
        dmodel = dask.delayed(inter_evaluator)
        predictions = [predict(task, dmodel) for task in tasks]
        predictions = dask.compute(*predictions)
        intermediates = {inter.code: inter for inter in predictions}
        with open(ARGS.output+'_intermediates.pkl', "wb") as f:
            print("Saving intermediates to disk...")
            dump(intermediates, f)
    for rxn in rxns:
        rxn.update_intermediates(intermediates)
    ti = time()
    print(f"Total intermediate evaluation time: {ti - t0:.2f} s")

    # REACTION EVALUATION
    print(f"\nEnergy estimation of the {len(rxns)} reactions...")
    print("Reaction properties calculator: ", rxn_evaluator)
    if rxn_evaluator.device == "cuda" and rxn_evaluator.supports_batching:
        print(f"Evaluating in batches of {ARGS.batch_size_rxn}")
        batches = [rxns[i:i + ARGS.batch_size_rxn] for i in range(0, len(rxns), ARGS.batch_size_rxn)]
        for batch in tqdm(batches):
            rxn_evaluator(batch)
    else:
        results  = []
        tasks = [load_x(reaction) for reaction in rxns]
        dmodel = dask.delayed(rxn_evaluator)
        for i in range(0, len(tasks), ARGS.batch_size_rxn):
            batch_tasks = tasks[i:i+ARGS.batch_size_rxn]
            batch_predictions = [predict(t, dmodel) for t in batch_tasks]
            batch_results = dask.compute(*batch_predictions)
            with open(ARGS.output + '_reactions.pkl', 'ab') as f:
                for result in batch_results:
                    dump(result, f)
            del batch_predictions, batch_results
            gc.collect()
            print(f"Finalized batch {i//ARGS.batch_size_rxn + 1}/{(len(tasks)-1)//ARGS.batch_size_rxn + 1}")
    client.shutdown()
    client.close()
    cluster.close()
    results = []
    try:
        with open(ARGS.output + '_reactions.pkl', 'rb') as f:
            while True:
                results.append(load(f))
    except EOFError:
        pass
    rxns = sorted(results)
    tr = time()
    print(f"Total reaction evaluation time: {tr - ti:.2f} s")

    print(
                "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
            )
    crn = ReactionNetwork(reactions=rxns, surface=surface)

    print(f"Total time: {(time() - t0):.2f} s")
    with open(ARGS.output+'.pkl', "wb") as f:
        dump(crn, f)
        print(f"CRN saved to {ARGS.output+'.pkl'}")
    if os.path.exists(ARGS.output + '_intermediates.pkl'):
        os.remove(ARGS.output + '_intermediates.pkl')
    if os.path.exists(ARGS.output + '_reactions.pkl'):
        os.remove(ARGS.output + '_reactions.pkl')

if __name__ == '__main__':
    main()
