"""Adsorption/Desorption template"""

import multiprocessing as mp

from ase import Atoms
from numpy import array_split
from rich.progress import Progress

from care import ElementaryReaction, Intermediate


def gen_adsorption_reactions(
    intermediates: dict[str, Intermediate], num_processes=mp.cpu_count()
) -> list[ElementaryReaction]:
    """
    Generate adsorption/desorption reactions
    for the closed-shell species in the reaction network.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule, and each value is the corresponding Intermediate instance.
    surf_inter : Intermediate
        The Intermediate instance of the surface.
    num_processes : int
        The number of processes to use for parallelization.

    Returns
    -------
    adsorption_steps : list[ElementaryReaction]
        adsorption reactions of the reaction network as ElementaryReaction instances.
    """

    surf_inter = Intermediate.from_molecule(
        Atoms(), code="*", is_surface=True, phase="surf"
    )

    gas_intermediates = [
        inter for inter in intermediates.values() if inter.phase == "gas"
    ]

    inter_chunks = array_split(gas_intermediates, num_processes)

    manager_ads_rxn = mp.Manager()
    progress_queue_rxn = manager_ads_rxn.Queue()

    tasks = [(chunk, surf_inter, progress_queue_rxn) for chunk in inter_chunks]

    with mp.Pool(mp.cpu_count()) as pool:
        result_async = pool.starmap_async(process_ads_react_chunk, tasks)
        with Progress() as progress:
            task_desc = format_description("[green]Generating adsorption reactions...")
            task = progress.add_task(task_desc, total=len(tasks))
            processed_items = 0

            while not result_async.ready():
                while not progress_queue_rxn.empty():
                    progress_queue_rxn.get()
                    processed_items += 1
                    progress.update(task, advance=1)

    # Combine the results from all chunks
    adsorption_steps = list(
        set([rxn for sublist in result_async.get() for rxn in sublist])
    )

    # Dissociative adsorptions (H2 and O2)
    for molecule in ["UFHFLCQGNIYNRP-UHFFFAOYSA-N", "MYMOFIZGZYHOMD-UHFFFAOYSA-N"]:
        gas_code = molecule + "g"
        if molecule == "UFHFLCQGNIYNRP-UHFFFAOYSA-N":  # H2
            ads_code = "YZCKVEUIGOORGS-UHFFFAOYSA-N*"
        else:  # O2
            ads_code = "QVGXLLKOCUKJST-UHFFFAOYSA-N*"
        adsorption_steps.append(
            ElementaryReaction(
                components=(
                    frozenset([surf_inter, intermediates[gas_code]]),
                    frozenset([intermediates[ads_code]]),
                ),
                r_type="adsorption",
            )
        )

    return adsorption_steps


def format_description(description: str, width: int = 45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]


def process_ads_react_chunk(
    inter_chunk: list[Intermediate], surf_inter: Intermediate, progress_queue
) -> list[ElementaryReaction]:
    """
    Processes a chunk of the intermediates to generate the adsorption reactions as ElementaryReaction instances.

    Parameters
    ----------
    inter_chunk : list[Intermediate]
        A subset of the intermediates dictionary keys to process.
    surf_inter : Intermediate
        The Intermediate instance of the surface.

    Returns
    -------
    adsorption_steps : list[ElementaryReaction]
        List of all the adsorption reactions of the reaction network as ElementaryReaction instances.
    """
    adsorptions = []
    for inter in inter_chunk:
        ads_inter = Intermediate.from_molecule(
            inter.molecule, code=inter.code[:-1] + "*", phase="ads"
        )
        adsorptions.append(
            ElementaryReaction(
                components=(frozenset([surf_inter, inter]), frozenset([ads_inter])),
                r_type="adsorption",
            )
        )

    progress_queue.put(1)
    return adsorptions
