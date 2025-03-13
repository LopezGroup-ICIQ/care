"""
Interface to Open Catalyst Project (OCP) models.
"""

from ase.optimize import BFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface, ElementaryReaction
from care.crn.utils.electro import Electron, Proton, Water
from care.constants import K_B
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.adsorption import place_adsorbate

class OCPIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        name: str = 'EquiformerV2-31M-S2EF-OC20-All+MD',
        cpu: bool = True,
        fmax: float = 0.05,
        max_steps: int = 5,
        num_configs: int = 1,
        del_traj: bool = True,
        **kwargs
    ):
        """Interface for the models from the Open Catalyst Project
        (OCP) for predicting the energy of an intermediate on a surface.

        Args:

        surface (Surface): The surface on which the reaction network is adsorbed.
        name (str): The name of the model to use among the checkpoints available in fairchem (OC20 and OC22)
        cpu (bool): Whether to use the CPU for the calculation. Default is True
        fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
        max_steps (int): The maximum number of steps for the relaxation. Default is 100.
        num_configs (int): The number of configurations to consider for the adsorbed phase. Default to 1.
        del_traj (bool): If True, keep relaxation trajectory and calculator for each intermediate configuration; 
                         note that this option may imply 10e6x larger CRN files!

        Note:

        - The intermediate energy is stored as E_tot - E_slab in eV.
        """
        from fairchem.core.models.model_registry import model_name_to_local_file
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.model_name = name
        self.checkpoint_path = model_name_to_local_file(name, local_cache='/tmp/fairchem_checkpoints/')
        self.surface = surface
        self.calc = OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=cpu, seed=42)
        self.num_params = sum([p.numel() for p in self.calc.trainer.model.parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.eref = {'C': -7.282, 'H': -3.477, 'O': -7.204, 'N': -8.083}  # eV
        self.del_traj = del_traj

    def __repr__(self) -> str:
        return f'{self.model_name} from Meta fairchem models'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, Intermediate):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be an Intermediate object.")
        
    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']
    
    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]

    def eval(
        self,
        intermediate: Intermediate,
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """
        if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
            raise ValueError(
                f'OCP models can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain)} elements.'
            )
        
        gas_energy = intermediate['C']*self.eref['C'] + intermediate['H']*self.eref['H'] + intermediate['O']*self.eref['O'] + intermediate['N']*self.eref['N']

        if intermediate.phase == "gas":  # gas phase
            intermediate.ads_configs = {
                "gas": {
                    "ase": intermediate.molecule,
                    "mu": gas_energy,  # eV
                    "s": 0.0,  # eV
                }
            }
        elif intermediate.phase == "ads":  # adsorbed
            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, self.surface, self.num_configs)
            for i, adsorption in enumerate(adsorptions):
                adsorption.calc = self.calc
                opt = BFGS(adsorption)
                opt.run(fmax=self.fmax, steps=self.max_steps)
                ads_config_dict[str(i)] = {}
                ads_config_dict[str(i)]['ase'] = adsorption
                # Note: OCP output is Eads, so to get Etot - Eslab, we need to add the gas-phase energy of the adsorbate
                ads_config_dict[str(i)]['mu'] = adsorption.get_potential_energy() + gas_energy
                ads_config_dict[str(i)]['s'] = 0.0
                if self.del_traj:
                    adsorption.calc = None
            intermediate.ads_configs = ads_config_dict
            print(intermediate.ads_configs)
        else:
            raise ValueError("Phase not supported by the current estimator.")


class OCPReactionEvaluator(ReactionEnergyEstimator):
    def __init__(
        self,
        intermediates: dict[str, Intermediate],
        T: float = 298.0,
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        """
        For now, thermodynamic properties are only calculated, not for electro-purposes yet.
        Eact = Delta E if endothermic, 0 if exothermic.
        
        Args:
            intermediates (dict): Dictionary of intermediates already evaluated.
            T (float): Temperature in Kelvin. Required for electrochemical reactions. Defaults to 298 K.
            pH (float): pH of the system. Required for electrochemical reactions. Defaults to 7.
            U (float): Potential of the system. Required for electrochemical reactions. Defaults to 0 V.
        """

        self.intermediates = intermediates
        self.pH = pH
        self.U = U
        self.T = T

    def __repr__(self) -> str:
        return f'Barrierless reaction evaluator (no lateral interactions)'

    def __call__(self,
                rxn: ElementaryReaction) -> None:
        self.eval(rxn)

    def __call__(self,
                 rxn: ElementaryReaction) -> None:
        self.eval(rxn)

    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']

    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]

    def calc_reaction_energy(self, reaction: ElementaryReaction) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        mu_is, mu_fs = 0.0, 0.0        
        for reactant in reaction.reactants:
            if reactant.is_surface or isinstance(reactant, Electron):
                continue
            elif isinstance(reactant, (Water, Proton)):  # Electrochemical conditions
                reactant_formula = "H2O" if isinstance(reactant, Water) else "H2"
                gas_inter = [
                    inter
                    for inter in self.intermediates.values()
                    if inter.formula == reactant_formula and inter.phase == "gas"
                ][0]
                x = 0.5 if reactant_formula == "H2" else 1.0
                energy_list = [
                    config["mu"] * x for config in gas_inter.ads_configs.values()
                ]
            else:
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[reactant.code].ads_configs.values()
                ]
            e_min_config = min(energy_list)
            mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
        for product in reaction.products:
            if product.is_surface or isinstance(product, Electron):       
                continue
            elif isinstance(product, (Water, Proton)):  # Electrochemical conditions
                product_formula = "H2O" if isinstance(product, Water) else "H2"
                gas_inter = [
                    inter
                    for inter in self.intermediates.values()
                    if inter.formula == product_formula and inter.phase == "gas"
                ][0]
                x = 0.5 if product_formula == "H2" else 1.0
                energy_list = [
                    config["mu"] * x for config in gas_inter.ads_configs.values()
                ] 
            else:
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[product.code].ads_configs.values()
                ]
            e_min_config = min(energy_list)
            mu_fs += abs(reaction.stoic[product.code]) * e_min_config
        reaction.e_is = mu_is, 0.0
        reaction.e_fs = mu_fs, 0.0
        mu_rxn = mu_fs - mu_is
        if reaction.r_type == "PCET":
            mu_rxn -= reaction.stoic["e-"] * (self.U + 2.303 * K_B * self.T * self.pH)
        reaction.e_rxn = mu_rxn, 0.0

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
