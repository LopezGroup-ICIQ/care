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
from care.crn.intermediate import SurfaceSite
from care.evaluators import load_evaluator
from care.io import save_network, load_network
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
        help="Path to network blueprint file.",
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
    crn = load_network(ARGS.bp)
    
    if crn.is_evaluated:
        raise ValueError("The input CRN blueprint has already been evaluated. Please provide a blueprint that has not been evaluated yet.")

    # Load evaluation settings
    with open(ARGS.input, "rb") as f:
        config = tomllib.load(f)

    # Check on input toml entries
    if "surface" not in config.keys():
        raise KeyError("'surface' field definition not found in input .toml file. Please define the surface where you want to evaluate your CRN.")
    if "evaluator" not in config.keys():
        raise KeyError("'evaluator' field definition not found in the input .toml file. Please define the energy evaluator.")

    surface = load_surface(**config["surface"])
    crn.add_catalyst(surface)

    model_name = config["evaluator"]["model"]
    del config["evaluator"]["model"]
    ml_evaluator = load_evaluator(model_name, **config["evaluator"])
    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
        print(f"{LOGO}\n")

    # 2. Evaluation of the adsorbed intermediates in the CRN with GAME-Net-UQ
    print(f"\n┏━━━━━━━━━━━━ Evaluating CRN ━━━━━━━━━━━┓\n")
    t0 = time()
    print("Energy evaluator: ", print(ml_evaluator))
    
    # =========================================================================
    # INTERMEDIATE EVALUATION
    # =========================================================================
    print(f"Energy estimation of the {crn.num_intermediates} intermediates...")
    
    if os.path.exists(ARGS.output + "_intermediates.pkl"):
        print("Loading intermediates from disk...")
        with open(ARGS.output + "_intermediates.pkl", "rb") as f:
            intermediates = load(f)
    else:
        if ml_evaluator.device == "cuda":
            print("GPU detected: Processing intermediates sequentially.")
            intermediates = {}
            for inter in tqdm(crn.intermediates.values()):
                ml_evaluator(inter)
                intermediates[inter.code] = inter
        else:
            cluster = LocalCluster(n_workers=ARGS.num_cpu,
                                   threads_per_worker=1, 
                                   ip='127.0.0.1', 
                                   scheduler_port=0, 
                                   dashboard_address=":0")
            client = Client(address=cluster)
            print(f"Dask dashboard available at: {cluster.dashboard_link}")
            
            tasks = [load_x(intermediate) for intermediate in crn.intermediates.values()]
            dmodel = dask.delayed(ml_evaluator)
            predictions = [predict(task, dmodel) for task in tasks]
            predictions = dask.compute(*predictions)
            intermediates = {inter.code: inter for inter in predictions}
            
            client.shutdown()
            client.close()
            cluster.close()

        with open(ARGS.output+'_intermediates.pkl', "wb") as f:
            print("Saving intermediates to disk...")
            dump(intermediates, f)
        
    ti = time()
    print(f"Total intermediate evaluation time: {ti - t0:.2f} s")


    # =========================================================================
    # REACTION EVALUATION
    # =========================================================================
    print(f"\nEnergy estimation of the {crn.num_reactions} reactions...")

    if ml_evaluator.device == "cuda":
        rxns = []
        print("GPU detected: Processing reactions sequentially.")
        for rxn in tqdm(crn.reactions):
            ml_evaluator(rxn)
            rxns.append(rxn)
    else:
        cluster = LocalCluster(n_workers=ARGS.num_cpu,
                               threads_per_worker=1, 
                               ip='127.0.0.1', 
                               scheduler_port=0)
        client = Client(address=cluster)
        tasks = [load_x(reaction) for reaction in crn.reactions]
        dmodel = dask.delayed(ml_evaluator)
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
        
        # Collect from Dask pickle files
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

    print("\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
    
    crn = ReactionNetwork(reactions=rxns, surface=surface)

    print(f"Total time: {(time() - t0):.2f} s")
    save_network(crn, f"{ARGS.output}.json")
    
    if os.path.exists(ARGS.output + '_intermediates.pkl'):
        os.remove(ARGS.output + '_intermediates.pkl')
    if os.path.exists(ARGS.output + '_reactions.pkl'):
        os.remove(ARGS.output + '_reactions.pkl')
