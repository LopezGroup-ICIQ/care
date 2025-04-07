"""
Interface to ORB potentials.
"""

from copy import deepcopy

from ase.optimize import BFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface, ElementaryReaction
from care.crn.utils.electro import Electron, Proton, Water
from care.constants import K_B
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.adsorption import place_adsorbate

class ORBIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        version: str = "orb-v2",
        device: str = "cpu",
        fmax: float = 0.05,
        brute_force_knn: bool = None,
        radius: float = 10.0,
        max_num_neighbors: int = 20,
        max_steps: int = 100,
        dtype: str = "float32",
        num_configs: int = 1,
        del_traj: bool = True,
        **kwargs
    ):
        """Interface to the ORB potentials.

        Args:
            surface (Surface): The surface on which the reaction network is adsorbed.
            version (str): The version of the employed ORB potential. Default to orb-v2.
            device (str): The device to use for the calculation. Default is "cpu".
            fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
            brute_force_knn (bool): whether to use a 'brute force' k-nearest neighbors method for graph construction.
                Defaults to None, in which case brute_force is used if a GPU is available (2-6x faster), but not on CPU (1.5x faster - 4x slower). 
                For very large systems (>10k atoms), brute_force may OOM on GPU, so it is recommended to set to False in that case.
            radius (float): The radius to use for the k-nearest neighbors method. Default is 10.0.
            max_num_neighbors (int): The maximum number of neighbors to consider for the k-nearest neighbors method. Default is 20.
            max_steps (int): The maximum number of steps for the relaxation. Default is 100.
            dtype (str): The data type to use for the calculation. Default is "float32".
            num_configs (int): The number of configurations to consider for the adsorbed phase. Default is 1.
            del_traj (bool): If True, keep relaxation trajectory and calculator for each intermediate configuration; 
                             note that this option may imply 10e6x larger CRN files!
        """
        from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
        from orb_models.forcefield.calculator import ORBCalculator, SystemConfig

        if version not in ORB_PRETRAINED_MODELS:
            raise ValueError(f"Version {version} not existing. Choose from {list(ORB_PRETRAINED_MODELS.keys())}.")
        self.version = version
        self.surface = surface
        self.slab_energy = 0.0
        self.dtype = dtype
        self.device = device
        self.model = ORB_PRETRAINED_MODELS[version](device=device)
        self.calc = ORBCalculator(model=self.model, 
                                  brute_force_knn=brute_force_knn, 
                                  system_config=SystemConfig(radius=radius, max_num_neighbors=max_num_neighbors), 
                                  device=device)
        self.num_params = sum([p.numel() for p in self.calc.model.parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.del_traj = del_traj
        self.get_slab_energy()

    def __repr__(self) -> str:
        return f'ORB potential ({self.version}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, Intermediate):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be an Intermediate object.")

    def get_slab_energy(self):
        self.surface.slab.calc = self.calc
        opt = BFGS(self.surface.slab)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        self.slab_energy = self.surface.slab.get_potential_energy()
        if self.del_traj:
            self.surface.slab.calc = None
        self.surface.energy = self.slab_energy
        print('self.slab_energy: ', self.slab_energy)

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]

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
                f'ORB can only evaluate molecules with {", ".join(self.adsorbate_domain)} elements.'
            )
        if intermediate.phase == 'gas':  # gas
            molec_eval = deepcopy(intermediate.molecule)
            molec_eval.set_cell([10, 10, 10])  # TODO: Should be function of molecule size

            molec_eval.calc = self.calc
            opt = BFGS(molec_eval)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            intermediate.ads_configs = {
                intermediate.phase: {
                    "ase": molec_eval,
                    "mu": molec_eval.get_potential_energy(),  # eV
                    "s": 0.0,  # eV
                }
            }
            if self.del_traj:
                molec_eval.calc = None
            print(intermediate.ads_configs)
        elif intermediate.phase == "ads":  # adsorbed
            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, self.surface, self.num_configs)
            for i, adsorption in enumerate(adsorptions):
                ads_config_dict[str(i)] = {}
                adsorption.calc = self.calc
                opt = BFGS(adsorption)
                opt.run(fmax=self.fmax, steps=self.max_steps)
                ads_config_dict[str(i)]['ase'] = adsorption
                ads_config_dict[str(i)]['mu'] = adsorption.get_potential_energy() - self.slab_energy # eV
                ads_config_dict[str(i)]['s'] = 0.0
                if self.del_traj:
                    adsorption.calc = None
            intermediate.ads_configs = ads_config_dict
        else:
            raise ValueError("Phase not supported by the current estimator.")


class ORBReactionEvaluator(ReactionEnergyEstimator):
    def __init__(
        self,
        intermediates: dict[str, Intermediate], 
        T: float = 298.0,
        electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        """
        For now, thermodynamic properties are only calculated, not for electro-purposes yet.
        Eact = Delta E for endothermic reactions, 0 for exothermic ones.
        
        Args:
            intermediates (dict): Dictionary of intermediates already evaluated.
            T (float): Temperature in Kelvin. Required for electrochemical reactions. Defaults to 298 K.
            electrode (str): Electrode potential. Required for electrochemical reactions. It can be 
                                SHE (Standard Hydrogen Electrode) or RHE (Reversible Hydrogen Electrode).
                                Defaults to SHE. With RHE, T and pH are not required.
            pH (float): pH of the system. Required for electrochemical reactions. Defaults to 7.
            U (float): Potential of the system. Required for electrochemical reactions. Defaults to 0 V.
        """

        self.intermediates = intermediates
        self.pH = pH
        self.U = U
        self.T = T
        self.electrode = electrode
        if self.electrode not in ["SHE", "RHE"]:
            raise ValueError(
                f"Electrode potential must be SHE or RHE. {self.electrode} is not supported."
            )

    def __repr__(self) -> str:
        return f'Barrierless reaction evaluator (no lateral interactions)'

    def __call__(self,
                 rxn: ElementaryReaction) -> None:
        self.eval(rxn)

    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]

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
            if reactant.is_surface:
                continue
            elif isinstance(reactant, Electron):  # Electrochemical conditions
                mu_is += abs(reaction.stoic["e-"]) * (abs(reaction.stoic["e-"])*self.U + (1 if self.electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
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
            if product.is_surface:       
                continue
            elif isinstance(product, Electron):  # Electrochemical conditions
                mu_fs += abs(reaction.stoic["e-"]) * (abs(reaction.stoic["e-"])*self.U + (1 if self.electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
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
