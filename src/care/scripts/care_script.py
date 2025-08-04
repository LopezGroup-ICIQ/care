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
import time
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import dask
from dask.distributed import Client, LocalCluster

from care import ReactionNetwork, gen_blueprint, load_surface
from care.crn.utils.electro import Electron
from care.evaluators import load_inter_evaluator, load_reaction_evaluator, eval_dict


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
    ARGS = PARSER.parse_args()

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

        # 2. Evaluation of the adsorbed intermediates in the CRN with GAME-Net-UQ
        surface = load_surface(**config["surface"])
        print(
            f"\n┏━━━━━━━━━━━━ Evaluating the CRN on {surface}) ━━━━━━━━━━━┓\n"
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

        # 2.1 Intermediate evaluator
        print(" Energy estimation of the intermediates...")
        del config["evaluator"]["model"]
        inter_evaluator = load_inter_evaluator(model_name, surface, **config["evaluator"])
        print(" Intermediates energy calculator: ", inter_evaluator)

        cluster = LocalCluster(n_workers=ARGS.num_cpu, 
                           threads_per_worker=1)
        client = Client(address=cluster)
        print(client.dashboard_link)
        @dask.delayed
        def load_inter(inter):
            return inter
        tasks = [load_inter(intermediate) for intermediate in intermediates.values()]
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
            task = progress.add_task(" [green]Processing...", total=len(reactions))
            processed_items = 0
            for reaction in reactions:
                rxn_evaluator.eval(reaction)
                processed_items += 1
                progress.update(
                    task,
                    advance=1,
                    description=f" [green]Processing {processed_items}/{len(reactions)}...",
                )
        reactions = sorted(reactions)

        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

        for r in reactions:
            if Electron in r.reactants or Electron in r.products:
                crn_type = "electrochemical"
                break
        else:
            crn_type = "thermal"

        crn = ReactionNetwork(
            intermediates=intermediates,
            reactions=reactions,
            surface=surface,
            ncc=ncc,
            noc=noc,
            oc={"T": T, "P": P, "U": U, "pH": PH},
            type=crn_type,
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
            oc={"T": T, "P": P, "U": U, "pH": PH},
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
