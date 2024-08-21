import time
import warnings
import multiprocessing as mp

from prettytable import PrettyTable
from rdkit import RDLogger

from care import ElementaryReaction, Intermediate
from care.crn.templates import adsorption, pcet, rearrengement, dissociation, chemspace


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]


def gen_blueprint(
    ncc: int, noc: int, cyclic: bool, additional_rxns: bool, electro: bool, num_cpu: int = mp.cpu_count(), show_progress: bool = False
) -> tuple[dict[str, Intermediate], list[ElementaryReaction]]:
    """
    Generate the CRN blueprint by applying
    reaction templates to the chemical space defined by
    the input parameters.

    Parameters
    ----------
    ncc : int
        Network Carbon Cutoff, maximum number of C atoms in the intermediates
    noc : int
        Network Oxygen Cutoff, Maximum number of O atoms in the intermediates.
        if is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    cyclic : bool
        If True, generates cyclic compounds (epoxides).
    additional_rxns : bool
        If True, additional reactions are generated (rearrangement reactions).
    electro : bool
        If True, proton-coupled electron transfer reactions are generated.
    num_cpu : int, optional
        Number of CPU cores to use for the CRN generation, by default mp.cpu_count()
    show_progress : bool, optional
        If True, a progress bar is shown for each step of the blueprint generation, by default False

    Returns
    -------
    intermediates_dict : dict[str, Intermediate]
        Dictionary of the Intermediate instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule, and each value is a list of Intermediate instances for that molecule.
    rxns_list : list[ElementaryReaction]
        List of all the reactions of the reaction network as ElementaryReaction instances.
    """
    intermediates, reactions = {}, []

    table = PrettyTable()
    table.field_names = ["Category", "Number of Items", "Time (s)"]

    # Generate the chemical space (CS)
    t0cs = time.time()
    chemical_space = chemspace.gen_chemical_space(ncc, noc, cyclic, show_progress)
    ncs = len(chemical_space)
    tcs = time.time() - t0cs
    table.add_row(["Saturated molecules", ncs, f"{tcs:.2f}"])

    # Extend CS with dissociation reactions
    t0ecs = time.time()
    bb_inters, bb_steps = dissociation.gen_dissociation_reactions(chemical_space, num_cpu, show_progress)
    nfrags = len(bb_inters) - len(chemical_space)
    nbbsteps = len(bb_steps)
    intermediates.update(bb_inters)
    reactions.extend(bb_steps)
    tecs = time.time() - t0ecs
    trxn = tecs
    table.add_row(
        ["Fragments and unsaturated molecules", nfrags, f"{tecs:.2f}"], divider=True
    )
    table.add_row(["Dissociation reactions", nbbsteps, f""])

    # Adsorption/Desorption reactions
    t02 = time.time()
    ads_steps = adsorption.gen_adsorption_reactions(intermediates, num_cpu, show_progress)
    nadssteps = len(ads_steps)
    reactions.extend(ads_steps)
    t2 = time.time() - t02
    trxn += t2
    table.add_row(["Adsorption reactions", nadssteps, f"{t2:.2f}"])

    # (1,2)-H shift rearrangement reactions
    if additional_rxns:
        t03 = time.time()
        rearr_steps = rearrengement.gen_rearrangement_reactions(intermediates, num_cpu, show_progress)
        nrearrsteps = len(rearr_steps)
        reactions.extend(rearr_steps)
        t3 = time.time() - t03
        trxn += t3
        table.add_row(["Rearrangement reactions", nrearrsteps, f"{t3:.2f}"])

    # Proton-coupled electron transfer (PCET) reactions
    if electro:
        t04 = time.time()
        pcets = pcet.gen_pcet_reactions(intermediates, reactions, show_progress)
        npcets = len(pcets)
        reactions.extend(pcets)
        t4 = time.time() - t04
        trxn += t4
        table.add_row(["PCET reactions", npcets, f"{t4:.2f}"])

    table.add_row(["", "", ""], divider=True)
    table.add_row(["Total number of species", len(intermediates), f"{tcs+tecs:.2f}"])
    table.add_row(["Total number of reactions", len(reactions), f"{trxn:.2f}"])

    print(f"\n{table}")

    return intermediates, reactions
