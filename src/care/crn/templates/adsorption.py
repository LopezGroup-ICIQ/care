"""Adsorption/Desorption template"""

import multiprocessing as mp

from ase import Atoms
from numpy import array_split
import numpy as np
from rich.progress import Progress

from care import ElementaryReaction, Intermediate
from care.constants import K_B, K_BU


class Adsorption(ElementaryReaction):
    __slots__ = ("adsorbate_mass", "adsorbate")

    def __init__(self, components, r_type):
        super().__init__(components=components, r_type=r_type)
        self.adsorbate = [inter for inter in self.reactants if inter.phase == "gas"][0]
        self.adsorbate_mass = self.adsorbate.mass  # atomic mass units

    def reverse(self):
        self.__class__ = Desorption
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        self.r_type = "desorption"
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )
    
    def get_kinetic_constants(
        self, t: float, uq: bool = False, clip_eact: float = -1.0
    ) -> tuple:
        """
        Evaluate the kinetic constants of the reactions in the network
        with transition state theory and Hertz-Knudsen equation.

        Args:
            t (float): Temperature in Kelvin.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            clip_eact(bool, optional): If > 0.0, the activation energy will be clipped, only if 
                both the forward and reverse activation energies are > clip_eact.
                if zero, the reaction will be assumed to be barrierless.
        """
        e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1]) if uq else self.e_rxn[0]
        e_act = np.random.normal(self.e_act[0], self.e_act[1]) if uq else self.e_act[0]
        e_act_rev = e_act - e_rxn
        if isinstance(clip_eact, float):
            if clip_eact > 0.0 and e_act > 0 and e_act_rev > 0:
                if e_act > clip_eact and e_act_rev > clip_eact:
                    if e_act >= e_act_rev:
                        e_act = clip_eact + e_rxn
                    else:
                        e_act = clip_eact
            if clip_eact == 0.0:
                e_act = max(0.0, e_rxn)
        else:
            pass

        sticking_coeff = np.exp(-e_act/ K_B /t)  # unitless
        area_active_site = 1e-18  # m2
        adsorbate_mass_kg = self.adsorbate_mass * 1.66054e-27  # kg
        k_dir = area_active_site * sticking_coeff / (2 * np.pi * adsorbate_mass_kg * K_BU * t) ** 0.5
        k_eq = np.exp(-e_rxn / K_B / t)
        return k_dir, k_dir / k_eq

    def bb_order(self):
        """
        Set the reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        """
        self.reverse()


class Desorption(ElementaryReaction):
    __slots__ = ("adsorbate_mass", "adsorbate")

    def __init__(self, components, r_type):
        super().__init__(components=components, r_type=r_type)
        self.adsorbate = [inter for inter in self.products if inter.phase == "gas"][0]
        self.adsorbate_mass = self.adsorbate.mass  # atomic mass units

    def reverse(self):
        self.__class__ = Adsorption
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        self.r_type = "adsorption"
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )

    def bb_order(self):
        """
        Set the reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        """
        pass


def gen_adsorption_reactions(
    intermediates: dict[str, Intermediate], num_cpu: int=mp.cpu_count(), show_progress: bool=False
) -> list[Adsorption]:
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
    num_cpu : int
        The number of CPU cores to use for the generation of the adsorption reactions.
    show_progress : bool
        If True, a progress bar is shown for this step of the blueprint generation.

    Returns
    -------
    adsorption_steps : list[ElementaryReaction]
        adsorption reactions of the reaction network as ElementaryReaction instances.
    """

    surf_inter = Intermediate(code="*", molecule=Atoms(), is_surface=True, phase="surf")

    gas_intermediates = [
        inter for inter in intermediates.values() if inter.phase == "gas"
    ]

    inter_chunks = array_split(gas_intermediates, num_cpu)

    manager_ads_rxn = mp.Manager()
    progress_queue_rxn = manager_ads_rxn.Queue()

    tasks = [(chunk, surf_inter, progress_queue_rxn) for chunk in inter_chunks]

    if show_progress:
        with mp.Pool(num_cpu) as pool:
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
    else:
        with mp.Pool(num_cpu) as pool:
            result_async = pool.starmap_async(process_ads_react_chunk, tasks)
            while not result_async.ready():
                while not progress_queue_rxn.empty():
                    progress_queue_rxn.get()

    # Combine the results from all chunks
    adsorption_steps = list(
        set([rxn for sublist in result_async.get() for rxn in sublist])
    )

    # Dissociative adsorptions (H2, O2, N2)
    for molecule in ["UFHFLCQGNIYNRP-UHFFFAOYSA-N", "MYMOFIZGZYHOMD-UHFFFAOYSA-N", "IJGRMHOSHXDMSA-UHFFFAOYSA-N"]:
        if molecule+'*' not in intermediates.keys():
            continue
        else:
            if molecule == "UFHFLCQGNIYNRP-UHFFFAOYSA-N":  # H2
                ads_code = "YZCKVEUIGOORGS-UHFFFAOYSA-N*"  # H
            elif molecule == "MYMOFIZGZYHOMD-UHFFFAOYSA-N":  # O2
                ads_code = "QVGXLLKOCUKJST-UHFFFAOYSA-N*"  # O
            else:  # N2
                ads_code = "QJGQUHMNIGDVPM-UHFFFAOYSA-N*"  # N
            adsorption_steps.append(
                Adsorption(
                    components=(
                        frozenset([surf_inter, intermediates[molecule + "g"]]),
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
) -> list[Adsorption]:
    """
    Processes a chunk of the intermediates to generate the adsorption reactions as Adsorption instances.

    Parameters
    ----------
    inter_chunk : list[Intermediate]
        A subset of the intermediates dictionary keys to process.
    surf_inter : Intermediate
        The Intermediate instance of the surface.

    Returns
    -------
    adsorption_steps : list[Adsorption]
        List of all the adsorption reactions of the reaction network as Adsorption instances.
    """
    adsorptions = []
    for inter in inter_chunk:
        ads_inter = Intermediate(code=inter.code[:-1] + "*", molecule=inter.molecule, phase="ads")
        adsorptions.append(Adsorption(components=([surf_inter, inter], [ads_inter]), r_type="adsorption"))
    progress_queue.put(1)
    return adsorptions
