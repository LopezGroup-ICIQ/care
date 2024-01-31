import os
import toml
import multiprocessing as mp
from pickle import dump
import resource
from rich.progress import Progress
from prettytable import PrettyTable
import cpuinfo
import psutil
import time

from ase.db import connect

from care import MODEL_PATH, DB_PATH, Surface, ReactionNetwork
from care.constants import LOGO, METAL_STRUCT_DICT
from care.crn.utilities.chemspace import gen_chemical_space
from care.crn.reactors import DifferentialPFR, DynamicCSTR
from care.crn.visualize import write_dotgraph
from care.gnn.interface import GameNetUQInter, GameNetUQRxn

DFT_DB_PATH = '../src/care/data/FG2dataset.db'


def evaluate_intermediate(intermediate, model, progress_queue):
    """
    Evaluates an intermediate.

    Args:
        intermediate (Intermediate): The intermediate.
        model (GameNetUQ): The model.
    """
    eval_inter = model.estimate_energy(intermediate)
    progress_queue.put(1)
    return eval_inter


def evaluate_reaction(reaction, model, progress_queue):
    """
    Evaluates a reaction.

    Args:
        reaction (Reaction): The reaction.
        model (GameNetUQ): The model.
    """
    eval_rxn = model.estimate_energy(reaction)
    progress_queue.put(1)
    return eval_rxn


def main():
    total_time = time.time()
    # Load configuration file
    with open("config.toml", "r") as f:
        config = toml.load(f)

    print("\nInitializing CARE (Catalytic Automatic Reaction Estimator)...\n")
    print(f"{LOGO}\n")

    # Loading parameters
    ncc = config['chemspace']['ncc']
    noc = config['chemspace']['noc']
    # If noc is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    noc = noc if noc > 0 else ncc * 2 + 2

    # If noc is > ncc * 2 + 2, raise an error
    if noc > ncc * 2 + 2:
        raise ValueError("The noc value cannot be greater than ncc * 2 + 2.")

    metal = config['surface']['metal']
    hkl = config['surface']['hkl']

    # Loading surface from database
    metal_db = connect(os.path.abspath(DB_PATH))
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    surface_ase = metal_db.get_atoms(
        calc_type="surface", metal=metal, facet=metal_structure)
    surface = Surface(surface_ase, hkl)

    # 1. Generate the chemical space (chemical spieces and reactions)
    print(
        f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━ Generating the C{ncc}O{noc} Chemical Space  ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")

    intermediates, reactions = gen_chemical_space(ncc, noc)

    print("\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Chemical Space generated ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # 2. Evaluation of the chemical space
    # print(f"Evaluating the C{ncc}O{noc} chemical space on {metal}({hkl})\n")
    print(
        f"\n┏━━━━━━━━━━━━ Evaluating the C{ncc}O{noc} Chemical Space on {metal}({hkl}) ━━━━━━━━━━━┓\n")

    # 2.1. Adsorbate placement and energy estimation

    # 2.1.1. Intermediate energy estimation
    print(" Energy estimation of the intermediates...")
    intermediate_model = GameNetUQInter(MODEL_PATH, surface, DFT_DB_PATH)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    manager = mp.Manager()
    progress_queue = manager.Queue()

    tasks = [(intermediate, intermediate_model, progress_queue)
             for intermediate in intermediates.values()]

    with mp.Pool(mp.cpu_count()) as pool:

        result_async = pool.starmap_async(evaluate_intermediate, tasks)

        with Progress() as progress:
            task = progress.add_task(" [green]Processing...", total=len(tasks))
            processed_items = 0

            while not result_async.ready():
                while not progress_queue.empty():
                    progress_queue.get()
                    processed_items += 1
                    progress.update(
                        task, advance=1, description=f" Processing {processed_items}/{len(tasks)}")

    # Updating the intermediates with the estimated energies
    intermediates = {
        intermediate.code: intermediate for intermediate in result_async.get()}

    # 2.1.2. Reaction energy estimation
    print("\n Energy estimation of the reactions...")
    rxn_model = GameNetUQRxn(MODEL_PATH, intermediates)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    manager_rxn = mp.Manager()
    progress_queue_rxn = manager_rxn.Queue()

    tasks = [(reaction, rxn_model, progress_queue_rxn)
             for reaction in reactions]

    with mp.Pool(mp.cpu_count()) as pool:
        result_async = pool.starmap_async(evaluate_reaction, tasks)
        with Progress() as progress:
            task = progress.add_task(" [green]Processing...", total=len(tasks))
            processed_items = 0

            while not result_async.ready():
                while not progress_queue_rxn.empty():
                    progress_queue_rxn.get()
                    processed_items += 1
                    progress.update(
                        task, advance=1, description=f" Processing {processed_items}/{len(tasks)}")

    reactions = [reaction for reaction in result_async.get()]
    print("\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # 3. Building and saving the CRN
    crn = ReactionNetwork(intermediates, reactions)

    # Create output directory
    output_dir = f"C{ncc}O{noc}_{metal}{hkl}"
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving the CRN...")
    with open(f"{output_dir}/crn.pkl", "wb") as f:
        dump(crn, f)
    print("Done!")

    # 4. Running MKM
    if config['mkm']['run']:
        print("\nRunning the MKM...")
        oc = config['operating_conditions']
        y0 = config['initial_conditions']
        r = config['reactor']
        reactor = DifferentialPFR if r['type'] == 'DifferentialPFR' else DynamicCSTR 
        results = crn.run_microkinetic(y0, 
                                       oc['temperature'],
                                       oc['pressure'],
                                       model=reactor)
        print("\nSaving the MKM results...")
        with open(f"{output_dir}/mkm.pkl", "wb") as f:
            dump(results, f)

    write_dotgraph(results['run_graph'], f"{output_dir}/mkm_res.svg", 'CH2O2')

    ram_mem = psutil.virtual_memory().available / 1e9
    peak_memory_usage = (resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss) / 1e6

    table2 = PrettyTable()
    table2.field_names = ["Process", "Model", "Usage"]
    table2.add_row(
        ["Processor", f"{cpuinfo.get_cpu_info()['brand_raw']} ({mp.cpu_count()} cores)", f"{psutil.cpu_percent()}%"])
    table2.add_row(["RAM Memory", f"{ram_mem:.1f} GB available",
                   f"{peak_memory_usage / ram_mem * 100:.2f}% ({peak_memory_usage:.2f} GB)"], divider=True)
    table2.add_row(["Total Execution Time", "",
                   f"{time.time() - total_time:.2f}s"])

    print(f"\n{table2}")


if __name__ == "__main__":
    main()
