from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS

from care import Intermediate, Surface, ElementaryReaction
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import METALS
from care.evaluators.gamenet_uq.adsorption.placement import place_adsorbate

class OC20IntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        model_name: str = 'EquiformerV2-31M-S2EF-OC20-All+MD',
        cpu: bool = True,
        fmax: float = 0.05,
        max_steps: int = 100,
        num_configs: int = 1
    ):
        """Interface for the models from the Open Catalyst Project
        (OCP) for predicting the energy of an intermediate on a surface.

        Args:

        surface (Surface): The surface on which the reaction network is adsorbed.
        model_name (str): The name of the model to use among the oc20 models.
        cpu (bool): Whether to use the CPU for the calculation. Default is False.
        fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
        max_steps (int): The maximum number of steps for the relaxation. Default is 100.
        """

        self.model_name = model_name
        self.checkpoint_path = model_name_to_local_file(model_name, local_cache='/tmp/fairchem_checkpoints/')
        self.surface = surface
        self.calc = OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=cpu)
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.eref = {'C': -7.282, 'H': -3.477, 'O': -7.204}

    def __repr__(self) -> str:
        return f'{self.model_name} from OC20 models'

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
        gas_energy = intermediate['C']*self.eref['C'] + intermediate['H']*self.eref['H'] + intermediate['O']*self.eref['O']

        if intermediate.phase == "surf":  # active site
            intermediate.ads_configs = {
                "surf": {"ase": intermediate.molecule, "mu": 0.0, "s": 0.0}
            }
        elif intermediate.phase == "gas":  # gas phase
            intermediate.ads_configs = {
                "gas": {
                    "ase": intermediate.molecule,
                    "mu": gas_energy,  # eV
                    "s": 0.0,  # eV
                }
            }
        elif intermediate.phase == "ads":  # adsorbed
            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, self.surface)
            for i in range(self.num_configs):
                ads_config_dict[str(i)] = {}
                adsorptions[i].set_calculator(self.calc)
                opt = BFGS(adsorptions[i])
                opt.run(fmax=self.fmax, steps=self.max_steps)
                ads_config_dict[str(i)]['ase'] = adsorptions[i]
                ads_config_dict[str(i)]['mu'] = adsorptions[i].get_potential_energy() + gas_energy
                ads_config_dict[str(i)]['s'] = 0.0
            intermediate.ads_configs = ads_config_dict
            print(intermediate.ads_configs)
        else:
            raise ValueError("Phase not supported by the current estimator.")


class OC20ReactionEvaluator(ReactionEnergyEstimator):
    def __init__(
        self,
        intermediates: dict[str, Intermediate]
    ):
        """Evaluate TS with CaTTsunami based on OCP models.
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
        """
        self.calc_reaction_energy(reaction)
        reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        reaction.e_act = reaction.e_ts[0] - reaction.e_is[0], 0.0
