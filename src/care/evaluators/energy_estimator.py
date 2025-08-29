"""
Base class for energy estimators.
"""
from typing import Optional
from abc import ABC, abstractmethod
from typing import Union

from ase import Atoms

from care import Intermediate, Surface, ElementaryReaction
from care.crn.utils.electro import Electron, Proton, Water
from care.constants import K_B


class IntermediateEnergyEstimator(ABC):
    """
    Base class for intermediate energy estimators.
    """

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, inter: Intermediate, surf: Optional[Surface] = None) -> None:
        self.eval(inter, surf)

    @property
    @abstractmethod
    def adsorbate_domain(self) -> list[str]:
        """
        Define the adsorbate elements that the estimator can handle.
        """
        pass

    @property
    @abstractmethod
    def surface_domain(self) -> list[str]:
        """
        Define the surface elements that the estimator can handle.
        """
        pass


    @abstractmethod
    def eval(self, inter: Union[Atoms, Intermediate], surf: Optional[Surface] = None) -> None:
        """
        Estimate the energy of a state.

        Args:
            inter (Intermediate): The intermediate.
            surf (Surface, optional): The surface. Defaults to None.
        """
        pass


class ReactionEnergyEstimator(ABC):
    """
    Base class for reaction properties estimators.
    Base implementation considers all elementary reactions as barrierless, 
    i.e., Eact = Delta E if endothermic, 0 if exothermic.

    Args:
            intermediates (dict): Dictionary of intermediates already evaluated.
            T (float): Temperature in Kelvin. Required for electrochemical reactions. Defaults to 298 K.
            ref_electrode (str): Reference electrode required for electrochemical reactions. It can be 
                                "SHE" (Standard Hydrogen Electrode) or "RHE" (Reversible Hydrogen Electrode).
                                Defaults to "SHE". With "RHE", T and pH are not required.
            pH (float): pH of the system. Required for electrochemical reactions. Defaults to 7.
            U (float): Potential of the system. Required for electrochemical reactions. Defaults to 0 V.
    """

    @abstractmethod
    def __init__(
        self, 
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        self.pH = pH
        self.U = U
        self.T = T
        self.ref_electrode = ref_electrode
        self.supports_batching = False
        if self.ref_electrode not in ["SHE", "RHE"]:
            raise ValueError(
                f"Electrode potential must be SHE or RHE. {self.ref_electrode} is not supported."
            )
    
    def __call__(self, reaction: ElementaryReaction) -> None:
        self.eval(reaction)

    def __repr__(self) -> str:
        return f"Base-class reaction evaluator"

    @property
    @abstractmethod
    def adsorbate_domain(self) -> list[str]:
        """
        Define the adsorbate elements that the estimator can handle.
        """
        pass

    @property
    @abstractmethod
    def surface_domain(self) -> list[str]:
        """
        Define the surface elements that the estimator can handle.
        """
        pass

    def calc_reaction_energy(self, reaction: ElementaryReaction) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        mu_is, mu_fs = 0.0, 0.0        
        for species in list(reaction.reactants) + list(reaction.products):
            if species.is_surface:
                continue
            elif isinstance(species, Electron):  # Electrochemical conditions
                mu_is += abs(min(0, reaction.stoic["e-"])) * (abs(reaction.stoic["e-"])*self.U + (1 if self.ref_electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
                mu_fs += abs(max(0, reaction.stoic["e-"])) * (abs(reaction.stoic["e-"])*self.U + (1 if self.ref_electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
                continue
            elif isinstance(species, (Water, Proton)):  # Electrochemical conditions
                species_formula = "H2O" if isinstance(species, Water) else "H2"
                x = 0.5 if species_formula == "H2" else 1.0
                gas_inter = [
                    inter
                    for inter in reaction.extra_intermediates.values()
                    if inter.formula == species_formula and inter.phase == "gas"
                ][0]
                energy_list = [
                    config["mu"] * x for config in gas_inter.ads_configs.values()
                ]
            else:
                energy_list = [
                    config["mu"]
                    for config in species.ads_configs.values()
                ]
            e_min_config = min(energy_list)
            mu_is += abs(min(0, reaction.stoic[species.code])) * e_min_config
            mu_fs += abs(max(0, reaction.stoic[species.code])) * e_min_config
        reaction.e_is = mu_is, 0.0
        reaction.e_fs = mu_fs, 0.0
        reaction.e_rxn = mu_fs - mu_is, 0.0

    def eval(self, reaction: ElementaryReaction) -> None:
        """
        Estimate reaction properties. 
        This base implementation evaluates the elementary reactions as barrierless.

        Args:
            reaction (ElementaryReaction): The reaction.
        """
        for species in list(reaction.reactants) + list(reaction.products):
            if not species.is_surface and species.ads_configs == {}:
                raise ValueError(f"Species in {reaction.repr_hr} ElementaryReaction are not evaluated.")
        self.calc_reaction_energy(reaction)
        reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        reaction.e_act = reaction.e_ts[0] - reaction.e_is[0], 0.0
