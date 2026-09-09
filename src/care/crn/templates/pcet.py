"""Proton-coupled electron transfer (PCET) reaction template, implemented
according to the Computational Hydrogen Electrode (CHE) reference."""

from rich.progress import Progress

from care.constants import K_B

from care import ElementaryReaction, Intermediate
from care.crn.intermediate import SurfaceSite, GasSpecies
from care.crn.utils.electro import Proton, Electron, Water


class PCET(ElementaryReaction):
    """Class for proton-coupled electron transfer reactions."""
    
    __slots__ = (
        "alpha", 
        "_bader_energy", 
        "extra_intermediates", 
        "T", 
        "pH", 
        "U", 
        "ref_electrode"
    )

    def __init__(
        self, 
        components, 
        r_type, 
        stoic=None,
        extra_intermediates=None,
        T: float=298.0,
        pH: float=7.0,
        U: float=0.0,
        ref_electrode: str="SHE"
    ):
        super().__init__(components=components, r_type=r_type, stoic=stoic)
        self.alpha = 0.5  # charge transfer coefficient
        self._bader_energy = None
        
        # Thermodynamic state variables
        self.extra_intermediates = extra_intermediates or {}
        self.T = T
        self.pH = pH
        self.U = U
        self.ref_electrode = ref_electrode

    @property
    def bader_energy(self):
        if self._bader_energy is None:
            return self.energy
        return self._bader_energy

    @bader_energy.setter
    def bader_energy(self, other):
        self._bader_energy = other

    def bb_order(self):
        """
        Note: PCET electron transfer steps do not have an intrinsic bond-breaking direction.
        """
        if Proton() not in self.products:
            self.reverse()

    @property
    def e_is(self) -> float:
        """Dynamically calculates the initial state energy or returns explicit override."""
        if self._e_is is not None:
            return self._e_is

        for species in self.reactants:
            if self._get_species_energy(species) is None:
                return None
                
        return sum(
            abs(min(0, self.stoic[species.code])) * self._get_species_energy(species)
            for species in self.reactants
        )

    @e_is.setter
    def e_is(self, value: float):
        self._e_is = value
        self._e_rxn = None  
        self._e_act = None  

    @property
    def e_fs(self) -> float:
        """Dynamically calculates the final state energy or returns explicit override."""
        if self._e_fs is not None:
            return self._e_fs

        for species in self.products:
            if self._get_species_energy(species) is None:
                return None
                
        return sum(
            abs(max(0, self.stoic[species.code])) * self._get_species_energy(species)
            for species in self.products
        )

    @e_fs.setter
    def e_fs(self, value: float):
        self._e_fs = value
        self._e_rxn = None  
        self._e_act = None

    def _get_species_energy(self, species: Intermediate) -> float:
        """Helper method to extract CHE thermodynamic energy for a species."""
        if isinstance(species, SurfaceSite):
            return 0.0
            
        if isinstance(species, Electron):
            ph_correction = (1 if self.ref_electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH
            return -self.U + ph_correction
            
        if isinstance(species, (Water, Proton)):
            species_formula = "H2O" if isinstance(species, Water) else "H2"
            fraction = 0.5 if species_formula == "H2" else 1.0

            ref_gas = next(
                (inter for inter in self.extra_intermediates.values()
                if inter.formula == species_formula and isinstance(inter, GasSpecies)),
                None
            )
            
            if ref_gas is None or ref_gas.E is None:
                return None
                
            return ref_gas.E * fraction
            
        return getattr(species, 'E', None)

def gen_pcet_reactions(
    intermediates: dict[str, Intermediate], reactions: list[ElementaryReaction], show_progress: bool=False
) -> list[PCET]:
    """
    Generate the proton-coupled electron transfer reactions
    of the reaction network as ElementaryReaction instances.
    Computational Hydrogen Electrode (CHE) is used as the reference.

    Args:
    ----
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of the chemical space of the reaction network.
        Each key is the InChIKey of the molecule, and values are the corresponding Intermediate instance.
    reactions : list[ElementaryReaction]
        List of the reactions of the reaction network as ElementaryReaction instances.
    show_progress : bool, optional
        If True, a progress bar is shown for each step of the blueprint generation, by default False

    Returns:
    -------
    pcets : list[ElementaryReaction]
        List of the proton-coupled electron transfer reactions of the reaction network as ElementaryReaction instances.
    """

    pcets = []
    rtype = "PCET"

    oh_code = "TUJKJAMUKRIRHC-UHFFFAOYSA-N*"
    h_ads = [
        inter
        for inter in intermediates.values()
        if inter.formula == "H" and inter.phase == "ads"
    ][0]

    pcets.append(
        PCET(
            components=[[Proton(), Electron(), SurfaceSite()], [h_ads]], r_type=rtype
        )
    )  # H+ + e- + * -> H*

    if show_progress:
        with Progress() as progress:
            task_desc = format_description("[green]Generating PCET reactions...")
            task = progress.add_task(task_desc, total=len(reactions))

            for rxn in reactions:
                new_reactants, new_products = [], []
                if rxn.r_type in ("H-O", "C-H"):
                    for reactant in rxn.reactants:
                        if reactant.formula == "H":
                            new_reactants.extend([Proton(), Electron()])
                        else:
                            if not isinstance(reactant, SurfaceSite):
                                new_reactants.append(reactant)
                    for product in rxn.products:
                        if product.formula == "H":
                            new_products.extend([Proton(), Electron()])
                        else:
                            if not isinstance(product, SurfaceSite):
                                new_products.append(product)

                    pcets.append(
                        PCET(
                            components=[new_reactants, new_products], r_type=rtype
                        )
                    )
                elif rxn.r_type in ("C-O", "O-O"):
                    if oh_code in [inter.code for inter in rxn]:
                        for reactant in rxn.reactants:
                            if isinstance(reactant, SurfaceSite):
                                new_reactants.extend([Electron(), Proton()])
                            elif reactant.formula == "HO":
                                new_reactants.append(Water())
                            else:
                                new_reactants.append(reactant)
                        for product in rxn.products:
                            if isinstance(product, SurfaceSite):
                                new_products.extend([Electron(), Proton()])
                            elif product.formula == "HO":
                                new_products.append(Water())
                            else:
                                new_products.append(product)

                        pcets.append(
                            PCET(
                                components=[new_reactants, new_products], r_type="PCET"
                            )
                        )
                    else:
                        continue
                else:
                    pass
                progress.update(task, advance=1)
    else:
        for rxn in reactions:
            new_reactants, new_products = [], []
            if rxn.r_type in ("H-O", "C-H"):
                for reactant in rxn.reactants:
                    if reactant.formula == "H":
                        new_reactants.extend([Proton(), Electron()])
                    else:
                        if not isinstance(reactant, SurfaceSite):
                            new_reactants.append(reactant)
                for product in rxn.products:
                    if product.formula == "H":
                        new_products.extend([Proton(), Electron()])
                    else:
                        if not isinstance(product, SurfaceSite):
                            new_products.append(product)

                pcets.append(
                    PCET(
                        components=[new_reactants, new_products], r_type=rtype
                    )
                )
            elif rxn.r_type in ("C-O", "O-O"):
                if oh_code in [inter.code for inter in rxn]:
                    for reactant in rxn.reactants:
                        if isinstance(reactant, SurfaceSite):
                            new_reactants.extend([Electron(), Proton()])
                        elif reactant.formula == "HO":
                            new_reactants.append(Water())
                        else:
                            new_reactants.append(reactant)
                    for product in rxn.products:
                        if isinstance(product, SurfaceSite):
                            new_products.extend([Electron(), Proton()])
                        elif product.formula == "HO":
                            new_products.append(Water())
                        else:
                            new_products.append(product)

                    pcets.append(
                        PCET(
                            components=[new_reactants, new_products], r_type="PCET"
                        )
                    )
                else:
                    continue
            else:
                pass

    return pcets


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]
