"""
Interface to ORB potentials.
"""

from copy import deepcopy
from typing import Union

from ase import Atoms
from ase.optimize import BFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface
from care.evaluators import IntermediateEnergyEstimator
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data

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
        logfile: str = None,
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
            logfile (str): The path to the logfile for relaxation trajectories. Default is None. Use '-' for stdout.
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
        self.logfile = logfile
        self.is_mlp = True
        self.get_slab_energy()

    def __repr__(self) -> str:
        return f'ORB potential ({self.version}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, (Intermediate, Atoms)):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be a CARE Intermediate or ASE Atoms object.")

    def get_slab_energy(self):
        self.surface.slab.calc = self.calc
        opt = BFGS(self.surface.slab, 
                   logfile=self.logfile)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        self.slab_energy = self.surface.slab.get_potential_energy()
        if self.del_traj:
            self.surface.slab.calc = None
        self.surface.energy = self.slab_energy

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
        intermediate: Union[Intermediate, Atoms],
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """
        if isinstance(intermediate, Intermediate):
            if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
                raise ValueError(
                    f'ORB can only evaluate molecules with {", ".join(self.adsorbate_domain)} elements.'
                )
            if intermediate.phase == 'gas':  # gas
                molec_eval = deepcopy(intermediate.molecule)
                molec_eval.set_cell([10, 10, 10])  # TODO: Should be function of molecule size

                molec_eval.calc = self.calc
                opt = BFGS(molec_eval, 
                        logfile=self.logfile)
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
            elif intermediate.phase == "ads":  # adsorbed
                ads_config_dict = {}
                adsorptions = place_adsorbate(intermediate, self.surface, -1)
                for i, adsorption in enumerate(adsorptions):
                    if len(ads_config_dict) == self.num_configs or len(ads_config_dict) == len(adsorptions):
                        break
                    adsorption.calc = self.calc
                    opt = BFGS(adsorption,
                                logfile=self.logfile)
                    opt.run(fmax=self.fmax, steps=self.max_steps)
                    g = atoms_to_data(adsorption, adsorption.get_array("atom_tags"), -1, True)
                    if g is None:
                        continue
                    ads_config_dict[str(i)] = {}
                    ads_config_dict[str(i)]['ase'] = adsorption
                    ads_config_dict[str(i)]['mu'] = adsorption.get_potential_energy() - self.slab_energy # eV
                    ads_config_dict[str(i)]['s'] = 0.0
                    if self.del_traj:
                        adsorption.calc = None
                if len(ads_config_dict) == 0:
                    Warning(f"No valid adsorption configuration found for {intermediate.formula}, keep the last one.")
                    ads_config_dict["0"] = {}
                    ads_config_dict["0"]['ase'] = adsorption
                    ads_config_dict["0"]['mu'] = adsorption.get_potential_energy() - self.slab_energy # eV
                    ads_config_dict["0"]['s'] = 0.0
                    ads_config_dict["0"]['converged'] = opt.converged()
                    ads_config_dict["0"]['connectivity'] = False
                else:
                    intermediate.ads_configs = ads_config_dict
            else:
                raise ValueError("Phase not supported by the current estimator.")
        elif isinstance(intermediate, Atoms):
            intermediate.calc = self.calc
            opt = BFGS(intermediate,
                       logfile=self.logfile)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            if self.del_traj:
                intermediate.calc = None
        else:
            return NotImplementedError("Input must be an Intermediate or Atoms object.")
