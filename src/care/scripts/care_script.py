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
import time

from care import MODEL_PATH, DB_PATH, DFT_DB_PATH, ReactionNetwork, Intermediate
from care.constants import LOGO
from care.crn.utils.blueprint import gen_blueprint
from care.gnn.interface import GameNetUQInter, GameNetUQRxn
from care.utils import load_surface

lock = mp.Lock()


class MissingInputError(Exception):
    pass


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
        eval_inter[intermediate.code] = model.eval(intermediate)

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
    ARGS = PARSER.parse_args()

    if not ARGS.input:
        raise MissingInputError("An input TOML file is required to run this program.")

    total_time = time.time()

    # Load .toml configuration file
    with open(ARGS.input, "rb") as f:
        config = tomllib.load(f)

    print(f"{LOGO}\n")

    # Loading parameters
    ncc = config["chemspace"]["ncc"]
    noc = config["chemspace"]["noc"]
    cyclic = config["chemspace"]["cyclic"]
    additional_rxns = config["chemspace"]["additional"]
    electrochem = config["chemspace"]["electro"]
    crn_type = "electrochemical" if electrochem else "thermal"

    metal = config["surface"]["metal"]
    hkl = config["surface"]["hkl"]

    PH = config["operating_conditions"]["pH"] if electrochem else None
    U = config["operating_conditions"]["U"] if electrochem else None
    T = config["operating_conditions"]["temperature"]
    P = config["operating_conditions"]["pressure"]

    # Output directory
    OUTPUT_DIR = ARGS.output
    if OUTPUT_DIR is None:
        output_dir = f"C{ncc}O{noc}_{metal}{hkl}"
    else:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    crn_path = f"{output_dir}/crn.pkl"

    # 0. Check if the CRN already exists
    if (not os.path.exists(crn_path)) or (config["chemspace"]["regen"] == True):
        # 1. Generate CRN blueprint
        print(
            f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━ Generating the C{ncc}O{noc} CRN blueprint  ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        )

        intermediates, reactions = gen_blueprint(
            ncc, noc, cyclic, additional_rxns, electrochem
        )

        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CRN blueprint generated ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

        # 2. Evaluation of the adsorbed intermediates in the CRN
        print(
            f"\n┏━━━━━━━━━━━━ Evaluating the C{ncc}O{noc} CRN on {metal}({hkl}) ━━━━━━━━━━━┓\n"
        )

        # Load surface from ase db
        surface = load_surface(DB_PATH, metal, hkl)

        # 2.1 Intermediate evaluator
        print(" Energy estimation of the intermediates...")
        inter_evaluator = GameNetUQInter(MODEL_PATH, surface, DFT_DB_PATH)
        print(" Intermediates energy calculator: ", inter_evaluator)

        if inter_evaluator.db != None:
            print(" DFT database: ", DFT_DB_PATH)

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

        manager = mp.Manager()
        progress_queue = manager.Queue()

        if len(intermediates) < 10000:
            chunk_size = 1
            tasks = [[intermediate] for intermediate in intermediates.values()]
        else:
            chunk_size = len(intermediates) // (mp.cpu_count() * 10)
            tasks = [
                list(intermediates.values())[i : i + chunk_size]
                for i in range(0, len(intermediates), chunk_size)
            ]

        # Create empty folder to store tmp results
        tmp_folder = tempfile.mkdtemp()
        _, tmp_file = tempfile.mkstemp(suffix=".pkl", dir=tmp_folder)

        num_cpu = mp.cpu_count()

        with Progress() as progress:
            task = progress.add_task(" [green]Processing...", total=len(tasks))
            processed_items = 0

            with mp.Pool(num_cpu) as pool:
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

        # Check how many species were available in the DFT database
        if inter_evaluator.db != None:
            counter = 0
            for inter in intermediates.values():
                if "dft" in inter.ads_configs.keys():
                    counter += 1
            print(
                f"\n {counter}/{len(intermediates)} ({round(counter/len(intermediates)*100,1)}%) intermediates available in the DFT database."
            )

        # 2.2. Reaction evaluator
        print("\n Energy estimation of the reactions...")

        rxn_evaluator = GameNetUQRxn(MODEL_PATH, intermediates, T=T, U=U, pH=PH)
        print(" Reaction property calculator: ", rxn_evaluator)

        eval_reactions = []
        with Progress() as progress:
            task = progress.add_task(" [green]Processing...", total=len(reactions))
            processed_items = 0
            for reaction in reactions:
                eval_rxn = rxn_evaluator.eval(reaction)
                eval_reactions.append(eval_rxn)
                processed_items += 1
                progress.update(
                    task,
                    advance=1,
                    description=f" [green]Processing {processed_items}/{len(reactions)}...",
                )

        reactions = sorted(eval_reactions)

        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        )

        # 3. Building and saving the CRN
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

    # 4. Running MKM
    if config["mkm"]["run"]:
        print("\nRunning the microkinetic simulation...")
        results = crn.run_microkinetic(
            iv=config["initial_conditions"],
            oc={"T": T, "P": P, "U": U, "pH": PH},
            uq=config["mkm"]["uq"],
            nruns=config["mkm"]["uq_samples"],
            thermo=config["mkm"]["thermo"],
            solver=config["mkm"]["solver"],
            barrier_threshold=config["mkm"].get("barrier_threshold"),
            ss_tol=config["mkm"]["ss_tol"],
            tfin=config["mkm"]["tfin"],
            eapp=config["mkm"]["eapp"],
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
