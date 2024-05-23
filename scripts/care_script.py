import os
import toml
import multiprocessing as mp
from pickle import dump, load, dumps
import resource
from rich.progress import Progress
from prettytable import PrettyTable
import cpuinfo
import psutil
import tempfile
import time
import sys
sys.path.append('../src')

from ase.db import connect

from care import MODEL_PATH, DB_PATH, DFT_DB_PATH, Surface, ReactionNetwork, Intermediate
from care.constants import LOGO, METAL_STRUCT_DICT
from care.crn.utils.chemspace import gen_chemical_space
from care.gnn.interface import GameNetUQInter, GameNetUQRxn


lock = mp.Lock()


def evaluate_intermediate(chunk_intermediate: list[Intermediate], model, progress_queue, f_path):
    """
    Evaluates an intermediate.

    Args:
        intermediate (Intermediate): The intermediate.
        model (GameNetUQ): The model.
    """

    eval_inter = {}
    for intermediate in chunk_intermediate:
        eval_inter[intermediate.code] = model.eval(intermediate)

    progress_queue.put(1)

    with lock:
        with open(f_path, 'ab') as file:
            # write text to data
            pkl_str = dumps(eval_inter)
            file.write(pkl_str)


def main():
    total_time = time.time()
    # Load configuration file
    with open("template.toml", "r") as f:
        config = toml.load(f)

    print(f"{LOGO}\n")

    # Loading parameters
    ncc = config['chemspace']['ncc']
    noc = config['chemspace']['noc']   
    cyclic = config['chemspace']['cyclic']  
    additional_rxns = config['chemspace']['additional']
    metal = config['surface']['metal']
    hkl = config['surface']['hkl']

    # Load surface from database
    metal_db = connect(os.path.abspath(DB_PATH))
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    surface_ase = metal_db.get_atoms(
        calc_type="surface", metal=metal, facet=metal_structure)
    surface = Surface(surface_ase, hkl)

    # Electrochemical parameters
    electrochem = config['chemspace']['electro']
    crn_type = "electrochemical" if electrochem else "thermal"
    PH = config['operating_conditions']['pH'] if electrochem else None
    U = config['operating_conditions']['U'] if electrochem else None
    T = config['operating_conditions']['temperature'] 
    P = config['operating_conditions']['pressure'] 

    # Output directory
    output_dir = f"C{ncc}O{noc}_{metal}{hkl}"
    os.makedirs(output_dir, exist_ok=True)

    crn_path = f"{output_dir}/crn.pkl"

    # 0. Check if the CRN already exists
    if (not os.path.exists(crn_path)) or (config['chemspace']['regen'] == True):

        # 1. Generate the chemical space (chemical spieces and reactions)
        print(
            f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━ Generating the C{ncc}O{noc} Chemical Space  ━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        
        intermediates, reactions = gen_chemical_space(
            ncc, noc, cyclic, additional_rxns)
        crn = ReactionNetwork(intermediates=intermediates, 
                              reactions=reactions, 
                              surface=surface, 
                              ncc=ncc, 
                              noc=noc, 
                              type=crn_type)
        if electrochem:
            crn.set_electro()
        print(crn)
        print("\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Chemical Space generated ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

        # 2. Evaluation of the chemical space
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

        num_cpu = mp.cpu_count()
        if len(intermediates) < 10000:
            chunk_size = 1
            tasks = [[intermediate] for intermediate in intermediates.values()]
        else:
            chunk_size = len(intermediates) // (mp.cpu_count() * 10)
            tasks = [list(intermediates.values())[i:i + chunk_size]
                     for i in range(0, len(intermediates), chunk_size)]

        # Create empty folder to store temp results
        tmp_folder = tempfile.mkdtemp()
        _, tmp_file = tempfile.mkstemp(suffix='.pkl', dir=tmp_folder)

        with Progress() as progress:
            task = progress.add_task(
                " [green]Processing...", total=len(tasks))
            processed_items = 0

            with mp.Pool(num_cpu) as pool:
                pool.starmap(evaluate_intermediate, [
                            (task, intermediate_model, progress_queue, tmp_file) for task in tasks])
                
                while not progress_queue.empty():
                    progress.update(task, advance=progress_queue.get(), description=f" [green]Processing {processed_items}/{len(tasks)}...")
                    processed_items += 1

        intermediates = {}
        with open(tmp_file, 'rb') as file:
            while True:
                try:
                    intermediates.update(load(file))              
                except EOFError:
                    break

        # 2.1.2. Reaction energy estimation
        print("\n Energy estimation of the reactions...")
        rxn_model = GameNetUQRxn(MODEL_PATH, intermediates, T=T, U=U, pH=PH)

        eval_reactions = []
        with Progress() as progress:
            task = progress.add_task(
                " [green]Processing...", total=len(reactions))
            processed_items = 0
            for reaction in reactions:
                eval_rxn = rxn_model.eval(reaction)
                eval_reactions.append(eval_rxn)
                processed_items += 1
                progress.update(
                    task, advance=1, description=f" [green]Processing {processed_items}/{len(reactions)}...")

        reactions = sorted(eval_reactions)

        print(
            "\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation done ━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

        # 3. Building and saving the CRN
        crn = ReactionNetwork(intermediates=intermediates, 
                              reactions=reactions, 
                              surface=surface, 
                              ncc=ncc, 
                              noc=noc,
                              oc={'T': T, 'P': P, 'U': U, 'pH': PH},
                              type=crn_type)
        
        print("\nSaving the CRN...")
        with open(f"{output_dir}/crn.pkl", "wb") as f:
            dump(crn, f)
        print("Done!")

    else:
        print("Loading the CRN...")
        with open(crn_path, "rb") as f:
            crn = load(f)

    # 4. Running MKM
    if config['mkm']['run']:
        print("\nRunning the microkinetic simulation...") 
        results = crn.run_microkinetic(iv=config['initial_conditions'],
                                       oc={'T': T, 'P': P, 'U': U, 'pH': PH},
                                       uq=config['mkm']['uq'],
                                       nruns= config['mkm']['uq_samples'],
                                       thermo=config['mkm']['thermo'],
                                       solver=config['mkm']['solver'], 
                                       barrier_threshold=config['mkm'].get('barrier_threshold'), 
                                       ss_tol=config['mkm']['ss_tol'],
                                       tfin=config['mkm']['tfin'], 
                                       eapp=config['mkm']['eapp'],)
        
        print("\nSaving the microkinetic simulation...")

        with open(f"{output_dir}/mkm.pkl", "wb") as f:
            dump(results, f)

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
