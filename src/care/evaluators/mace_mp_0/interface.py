
from ase.optimize import BFGS
from mace.calculators import mace_mp

from care import Intermediate, Surface, ElementaryReaction
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import METALS
from care.evaluators.gamenet_uq.adsorption.placement import place_adsorbate

class MaceIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        size: str = "small",
        device: str = "cpu",
        fmax: float = 0.05,
        max_steps: int = 100,
        dtype: str = "float64",
        num_configs: int = 1,
        **kwargs
    ):
        """Interface for the MACE-MP-0 model.

        Args:

        surface (Surface): The surface on which the reaction network is adsorbed.
        size (str): The size of the model to use among the mace models. Default is "small".
        device (str): The device to use for the calculation. Default is "cpu".
        cpu (bool): Whether to use the CPU for the calculation. Default is False.
        fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
        max_steps (int): The maximum number of steps for the relaxation. Default is 100.
        dtype (str): The data type to use for the calculation. Default is "float64".
        num_configs (int): The number of configurations to consider for the adsorbed phase. Default is 1.
        """

        self.surface = surface
        self.size = size
        self.dtype = dtype
        self.device = device
        self.calc = mace_mp(model=self.size, device=self.device, default_dtype=dtype)
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs

    def __repr__(self) -> str:
        return f'MACE-MP-0 potential ({self.size}, {self.device}, {self.dtype})'


    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']

    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return METALS

    def eval(
        self,
        intermediate: Intermediate,
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """

        if intermediate.phase in ("surf", "gas"):  # active site
            intermediate.molecule.set_calculator(self.calc)
            opt = BFGS(intermediate.molecule)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            intermediate.ads_configs = {
                intermediate.phase: {
                    "ase": intermediate.molecule,
                    "mu": intermediate.molecule.get_potential_energy(),  # eV
                    "s": 0.0,  # eV
                }
            }
            print(intermediate.ads_configs)
        elif intermediate.phase == "ads":  # adsorbed
            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, self.surface)
            for i in range(self.num_configs):
                ads_config_dict[str(i)] = {}
                adsorptions[i].set_calculator(self.calc)
                opt = BFGS(adsorptions[i])
                opt.run(fmax=self.fmax, steps=self.max_steps)
                ads_config_dict[str(i)]['ase'] = adsorptions[i]
                ads_config_dict[str(i)]['mu'] = adsorptions[i].get_potential_energy()  # eV
                ads_config_dict[str(i)]['s'] = 0.0
            intermediate.ads_configs = ads_config_dict
            print(intermediate.ads_configs)
        else:
            raise ValueError("Phase not supported by the current estimator.")


class MaceReactionEvaluator(ReactionEnergyEstimator):
    def __init__(
        self,
        intermediates: dict[str, Intermediate], **kwargs
    ):
        """
        For now, thermodynamic properties are only calculated, not for electro-purposes yet.
        """

        self.intermediates = intermediates

    def __repr__(self) -> str:
        return f'Barrierless reaction evaluator (no lateral interactions)'

    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']

    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return METALS

    def calc_reaction_energy(self, reaction: ElementaryReaction) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        mu_is, mu_fs = 0.0, 0.0
        for reactant in reaction.reactants:
            if reactant.is_surface:
                continue
            energy_list = [
                config["mu"]
                for config in self.intermediates[reactant.code].ads_configs.values()
            ]
            e_min_config = min(energy_list)
            mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
        for product in reaction.products:
            if product.is_surface:
                continue
            energy_list = [
                config["mu"]
                for config in self.intermediates[product.code].ads_configs.values()
            ]
            e_min_config = min(energy_list)
            mu_fs += abs(reaction.stoic[product.code]) * e_min_config
        reaction.e_is = mu_is, 0.0
        reaction.e_fs = mu_fs, 0.0
        reaction.e_rxn = mu_fs - mu_is, 0.0

    def eval(
        self,
        reaction: ElementaryReaction
    ):

        """
        Given the reaction, return the properties of the reaction as attributes of the reaction object.
        For now, CatTsunami not implemented yet.
        """

        self.calc_reaction_energy(reaction)
        reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        reaction.e_act = reaction.e_ts[0] - reaction.e_is[0], 0.0
