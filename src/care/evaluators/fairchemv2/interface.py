"""
Interface to FairChemV2 UMA models.
"""
from copy import deepcopy
import logging
from typing import Union
import warnings

from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface
from care.evaluators import IntermediateEnergyEstimator
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    FAIRCHEMV2_AVAILABLE = True
except ImportError:
    FAIRCHEMV2_AVAILABLE = False

class FairChemV2IntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface = None,
        name: str = 'uma-s-1p1',
        device: str = 'cpu',
        task_name: str = 'oc20',
        fmax: float = 0.05,
        max_steps: int = 5,
        num_configs: int = 1,
        del_traj: bool = True,
        optimizer: str = 'BFGS',
        logfile: str = None,
        patience: int = 3,
        **kwargs
    ):
        """
        Interface to FairChemV2 UMA model. 
        To use this class of models, you need to have the permissions to access the fairchem models from Meta.

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
        if not FAIRCHEMV2_AVAILABLE:
            raise ImportError("The FairChemV2IntermediateEvaluator requires 'fairchem-core' to be installed. "
                "Please install it using pip install fairchem-core==2.19.0.")

        self.model_name = name
        self.surface = surface
        self.n_slab = len(surface.slab)  # required in cases when slab is resized to fit larger adsorbates
        self.device = device
        self.task_name = task_name
        self.predictor = pretrained_mlip.get_predict_unit(name, device=device)
        self.calc = FAIRChemCalculator(self.predictor, task_name=task_name)
        self.num_params = sum([p.numel() for p in self.calc.predictor.model.parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.patience = num_configs * patience
        self.is_mlp = True
        self.del_traj = del_traj
        self.optimizer = BFGS if optimizer == 'BFGS' else LBFGS
        self.logfile = logfile
        if self.surface is not None:
            self.get_slab_energy()

    def __repr__(self) -> str:
        return f'{self.model_name} from Meta FairChem-v2 models'

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
        
    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']
    
    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]
    
    def get_calculator(self):
        return FAIRChemCalculator(self.predictor, task_name=self.task_name)

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
                    f'UMA models can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain)} elements.'
                )

            if intermediate.phase == "gas":  # gas phase
                molec_eval = deepcopy(intermediate.molecule)
                molec_eval.set_cell([10, 10, 10])  # TODO: Should be function of molecule size

                molec_eval.calc = self.calc
                opt = BFGS(molec_eval, 
                        logfile=None)
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
                attempts = 0
                best_broken_ads = None
                lowest_broken_mu = float('inf')
                for i, adsorption in enumerate(adsorptions):
                    if len(ads_config_dict) == self.num_configs or attempts >= self.patience:
                        break
                    attempts += 1
                    adsorption.calc = self.calc
                    opt = self.optimizer(adsorption,
                            logfile=self.logfile)
                    opt.run(fmax=self.fmax, steps=self.max_steps)
                    current_energy = adsorption.get_potential_energy() - ((len(adsorption) - len(intermediate.molecule))/self.n_slab) * self.slab_energy
                    g = atoms_to_data(adsorption, adsorption.get_array("atom_tags"), -1, True)
                    if g is None:
                        if current_energy < lowest_broken_mu:
                            lowest_broken_mu = current_energy
                            best_broken_ads = adsorption.copy()
                        continue
                    ads_config_dict[str(i)] = {
                        'ase': adsorption,
                        'mu': current_energy,
                        's': 0.0,
                        'connectivity': True,
                        # 'converged': opt.converged()
                    }
                    if self.del_traj:
                        adsorption.calc = None
                if len(ads_config_dict) == 0:
                    warnings.warn(
                        f"Failed to find intact configuration for {intermediate.formula} "
                        f"after {attempts} attempts. Falling back to the lowest energy broken structure."
                    )
                    fallback_ads = best_broken_ads if best_broken_ads is not None else adsorptions[-1]
                    
                    ads_config_dict["0"] = {
                        'ase': fallback_ads,
                        'mu': (lowest_broken_mu if best_broken_ads is not None else fallback_ads.get_potential_energy()) - self.slab_energy,
                        's': 0.0,
                        'connectivity': False
                    }
                elif len(ads_config_dict) < self.num_configs:
                    logging.info(
                        f"Requested {self.num_configs} configs for {intermediate.formula}, "
                        f"but only found {len(ads_config_dict)} valid ones before hitting the attempt limit."
                    )

                intermediate.ads_configs = ads_config_dict
            else:
                raise ValueError("Phase not supported by the current estimator.")
        elif isinstance(intermediate, Atoms):
            intermediate.calc = self.calc
            opt = self.optimizer(intermediate,
                       logfile=None)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            if self.del_traj:
                intermediate.calc = None
        else:
            return NotImplementedError("Input must be an Intermediate or Atoms object.")