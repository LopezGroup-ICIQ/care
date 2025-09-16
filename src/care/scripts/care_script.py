import argparse
import os
import gc
import tomllib
import multiprocessing as mp
from pickle import dump, load
import resource
from prettytable import PrettyTable
import cpuinfo
import psutil
import time
from tqdm import tqdm

import dask
from dask.distributed import Client, LocalCluster

from care import ReactionNetwork, gen_blueprint, load_surface
from care.crn.utils.electro import Electron
from care.evaluators import load_inter_evaluator, load_reaction_evaluator, eval_dict
from care.scripts import setup_logging, load_x, predict

def main():
    """
    Parse .toml configuration file and run the CARE pipeline.
    """

    PARSER = argparse.ArgumentParser(
        description="CARE main script to generate and evaluate chemical reaction networks."
    )
    PARSER.add_argument(
        "-i",
        "--input",
        type=str,
        dest="input",
        help="Path to the .toml configuration file.",
    )
    PARSER.add_argument(
        "-o", "--output", type=str, dest="output", help="Path to the output directory."
    )
    PARSER.add_argument(
        "-ncpu",
        "--num_cpu",
        type=int,
        dest="num_cpu",
        help="Number of CPU cores to use for the CRN generation.",
        default=mp.cpu_count(),
    )
    PARSER.add_argument(
        '--log', 
        type=str, 
        help='Path to run log file', 
        default="care.log"
    )
    PARSER.add_argument(
        "-bs_rxn",
        type=int,
        dest="batch_size_rxn",
        help="Batch size for reaction evaluation. Default is 256.",
        default=512,
    )
    ARGS = PARSER.parse_args()
    setup_logging(ARGS.log)

    if not ARGS.input:
        raise ValueError("Input .toml file not provided.")

    total_time = time.time()
    with open(ARGS.input, "rb") as f:
        config = tomllib.load(f)

    BP_SWITCH, EVAL_SWITCH, MKM_SWITCH = False, False, False

    if "chemspace" in config.keys():
        BP_SWITCH = True
    if "surface" in config.keys() and "evaluator" in config.keys():
        EVAL_SWITCH = True
    if "mkm" in config.keys() and "operating_conditions" in config.keys() and "initial_conditions" in config.keys():
        MKM_SWITCH = True

    current_dir = os.path.dirname(__file__)
    logo_path = current_dir + "/../logo.txt"
    with open(logo_path, "r") as file:
        LOGO = file.read()
    print(f"{LOGO}\n")

    # Loading parameters
    ncc = config["chemspace"]["ncc"] if "ncc" in config["chemspace"] else None
    noc = config["chemspace"]["noc"] if "noc" in config["chemspace"] else None
    cs = config["chemspace"]["cs"] if "cs" in config["chemspace"] else None
    cyclic = config["chemspace"]["cyclic"] if "cyclic" in config["chemspace"] else None
    additional_rxns = config["chemspace"]["additional"] if "additional" in config["chemspace"] else None
    electrochem = config["chemspace"]["electro"] if "electro" in config["chemspace"] else None
    crn_type = "electrochemical" if electrochem else "thermal"

    PH = config["operating_conditions"]["pH"] if electrochem else None
    U = config["operating_conditions"]["U"] if electrochem else None
    T = config["operating_conditions"]["temperature"] if "operating_conditions" in config else None
    P = config["operating_conditions"]["pressure"] if "operating_conditions" in config else None

    # Output directory
    OUTPUT_DIR = ARGS.output
    if OUTPUT_DIR is None:
        output_dir = "crn_output"
    else:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=False)
    crn_path = f"{output_dir}/crn.pkl"

    # 0. Check if the CRN already exists
    if (not os.path.exists(crn_path)) or (config["chemspace"]["regen"] == True):
        print(
        f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Generating the CRN blueprint  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        )
        if ncc and not cs:
            print(f"ncc={ncc}, noc={noc}")
        else:
            print("Input chemical space (SMILES): {}".format(", ".join(cs)))

        intermediates, reactions = gen_blueprint(
                                            ncc=ncc,
                                            noc=noc,
                                            cs=cs,
                                            cyclic=cyclic,
                                            additional_rxns=additional_rxns,
                                            electro=electrochem,
                                            num_cpu=ARGS.num_cpu,
                                            show_progress=True
                                        )

        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CRN blueprint generated ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

        # 2. Evaluation of the adsorbed intermediates in the CRN
        surface = load_surface(**config["surface"])
        print(
            f"\n┏━━━━━━━━━━━━ Evaluating the CRN on {surface} ━━━━━━━━━━━┓\n"
        )

        # Check correct energy evaluator definition
        if "evaluator" not in config:
            raise ValueError("Evaluator model not defined in the input file.")
        else:
            if "model" not in config["evaluator"]:
                raise ValueError("Evaluator model not defined in the input file.")
            if config["evaluator"]["model"] not in eval_dict.keys():
                raise ValueError(
                    f"Model {config['evaluator']['model']} not found in the available evaluators {eval_dict}."
                )

        model_name = config["evaluator"]["model"]

        t0 = time.time()
        # 2.1 Intermediate evaluator
        print(f"Energy estimation of the {len(intermediates)} intermediates...")
        inter_evaluator = load_inter_evaluator(model_name, surface, **config["evaluator"])
        print("Intermediates energy calculator: ", inter_evaluator)
        del config["evaluator"]["model"]
        rxn_evaluator = load_reaction_evaluator(model_name, inter_evaluator, **config["evaluator"])

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
            tasks = [load_x(intermediate)
                for intermediate in intermediates.values()
            ]
            dmodel = dask.delayed(inter_evaluator)
            predictions = [predict(task, dmodel)
                for task in tasks
            ]
            predictions = dask.compute(*predictions)
            intermediates = {inter.code: inter for inter in predictions}
            with open(ARGS.output+'_intermediates.pkl', "wb") as f:
                print("Saving intermediates to disk...")
                dump(intermediates, f)
        for rxn in reactions:
            rxn.update_intermediates(intermediates)
        ti = time.time()
        print(f"Total intermediate evaluation time: {ti - t0:.2f} s")

        # REACTION EVALUATION
        print(f"\nEnergy estimation of the {len(reactions)} reactions...")
        print("Reaction properties calculator: ", rxn_evaluator)
        if rxn_evaluator.device == "cuda" and rxn_evaluator.supports_batching:
            print(f"Evaluating in batches of {ARGS.batch_size_rxn}")
            batches = [reactions[i:i + ARGS.batch_size_rxn] for i in range(0, len(reactions), ARGS.batch_size_rxn)]
            for batch in tqdm(batches):
                rxn_evaluator(batch)
        else:
            results  = []
            tasks = [load_x(reaction) for reaction in reactions]
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
        reactions = sorted(results)
        tr = time()
        print(f"Total reaction evaluation time: {tr - ti:.2f} s")


        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

        crn = ReactionNetwork(
            reactions=reactions,
            surface=surface,
            oc={"T": T, "P": P, "U": U, "pH": PH},
        )

        print("\nSaving the CRN...")
        with open(f"{output_dir}/crn.pkl", "wb") as f:
            dump(crn, f)
        print("Done!")

    else:
        print("Loading the CRN...")
        with open(crn_path, "rb") as f:
            crn = load(f)

    if MKM_SWITCH:
        print("\nRunning the microkinetic simulation...")
        results = crn.run_microkinetic(
            iv=config["initial_conditions"],
            oc=config["operating_conditions"],
            **config["mkm"]
        )

        print("\nSaving the microkinetic simulation...")

        with open(f"{output_dir}/mkm.pkl", "wb") as f:
            dump(results, f)

    ram_mem = psutil.virtual_memory().available / 1e9
    peak_memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e6

    table2 = PrettyTable()
    table2.field_names = ["Process", "Model", "Usage"]
    table2.add_row(
        [
            "Processor",
            f"{cpuinfo.get_cpu_info()['brand_raw']} ({mp.cpu_count()} cores)",
            f"{psutil.cpu_percent()}%",
        ]
    )
    table2.add_row(
        [
            "RAM Memory",
            f"{ram_mem:.1f} GB available",
            f"{peak_memory_usage / ram_mem * 100:.2f}% ({peak_memory_usage:.2f} GB)",
        ],
        divider=True,
    )
    table2.add_row(["Total Execution Time", "", f"{time.time() - total_time:.2f}s"])

    print(f"\n{table2}")
